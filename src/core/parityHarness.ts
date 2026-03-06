import { createHash } from 'node:crypto';
import path from 'node:path';
import { createDefaultReapMlxComponents } from './reapComponents.js';
import { runReapMlx } from './reapMlx.js';
import {
  ensureSecureDir,
  resolveSafePath,
  writeJsonAtomicSafe,
  writeTextAtomicSafe
} from './security.js';
import type { ExpertSignal, ModelTelemetry, PruneMethod, PruningPlan, RunConfig } from './types.js';

export interface ParityHarnessConfig {
  leftModelPath: string;
  rightModelPath: string;
  outputDir: string;
  targetRatio?: number;
  calibrationRounds?: number;
  minExpertsPerLayer?: number;
  pruneMethod?: PruneMethod;
  nExpertsToPrunePerLayer?: number;
  preserveSuperExperts?: boolean;
  preserveOutliers?: boolean;
  allowLegacySaliency?: boolean;
  requireIdenticalTelemetry?: boolean;
}

export interface ExpertSignalDiff {
  layer: number;
  expert: number;
  left?: ExpertSignal;
  right?: ExpertSignal;
}

export interface PruneDiffByLayer {
  layer: number;
  onlyLeft: number[];
  onlyRight: number[];
}

export interface ParityReport {
  version: '1.0';
  createdAt: string;
  config: {
    leftModelPath: string;
    rightModelPath: string;
    outputDir: string;
    targetRatio?: number;
    calibrationRounds?: number;
    minExpertsPerLayer?: number;
    pruneMethod: PruneMethod;
    nExpertsToPrunePerLayer?: number;
    preserveSuperExperts: boolean;
    preserveOutliers: boolean;
    allowLegacySaliency: boolean;
    requireIdenticalTelemetry: boolean;
  };
  leftTelemetry: {
    modelName: string;
    expertCount: number;
    metadata?: ModelTelemetry['metadata'];
    normalizedHash: string;
  };
  rightTelemetry: {
    modelName: string;
    expertCount: number;
    metadata?: ModelTelemetry['metadata'];
    normalizedHash: string;
  };
  telemetry: {
    modelNameEqual: boolean;
    metadataEqual: boolean;
    expertCountEqual: boolean;
    normalizedHashEqual: boolean;
    differingExpertRows: number;
    firstDifferingExpert?: ExpertSignalDiff;
  };
  pruning: {
    thresholdDelta: number;
    thresholdEqual: boolean;
    prunedExactMatch: boolean;
    keptExactMatch: boolean;
    onlyLeftPruned: Array<{ layer: number; expert: number }>;
    onlyRightPruned: Array<{ layer: number; expert: number }>;
    diffByLayer: PruneDiffByLayer[];
  };
  overallExactMatch: boolean;
  artifacts: {
    leftPlanPath: string;
    rightPlanPath: string;
    reportJsonPath: string;
    reportMarkdownPath: string;
  };
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((entry) => stableValue(entry));
  }

  if (value && typeof value === 'object') {
    const candidate = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(candidate)
        .sort((left, right) => left.localeCompare(right))
        .map((key) => [key, stableValue(candidate[key])])
    );
  }

  return value;
}

function stableStringify(value: unknown): string {
  return JSON.stringify(stableValue(value));
}

function hashNormalized(value: unknown): string {
  return createHash('sha256').update(stableStringify(value)).digest('hex');
}

function normalizeExpertSignal(signal: ExpertSignal): ExpertSignal {
  return stableValue(signal) as ExpertSignal;
}

function sortExperts(experts: readonly ExpertSignal[]): ExpertSignal[] {
  return [...experts]
    .map((entry) => normalizeExpertSignal(entry))
    .sort((left, right) => {
      if (left.layer !== right.layer) {
        return left.layer - right.layer;
      }

      return left.expert - right.expert;
    });
}

function normalizeTelemetry(telemetry: ModelTelemetry): ModelTelemetry {
  return {
    modelName: telemetry.modelName,
    experts: sortExperts(telemetry.experts),
    ...(telemetry.metadata
      ? { metadata: stableValue(telemetry.metadata) as NonNullable<ModelTelemetry['metadata']> }
      : {})
  };
}

function expertKey(layer: number, expert: number): string {
  return `${layer}:${expert}`;
}

function toExpertMap(experts: readonly ExpertSignal[]): Map<string, ExpertSignal> {
  return new Map(
    sortExperts(experts).map((entry) => [expertKey(entry.layer, entry.expert), entry] as const)
  );
}

function diffExperts(
  leftExperts: readonly ExpertSignal[],
  rightExperts: readonly ExpertSignal[]
): {
  differingRows: number;
  firstDifferingExpert?: ExpertSignalDiff;
} {
  const leftMap = toExpertMap(leftExperts);
  const rightMap = toExpertMap(rightExperts);
  const keys = [...new Set([...leftMap.keys(), ...rightMap.keys()])].sort((left, right) =>
    left.localeCompare(right, undefined, { numeric: true })
  );

  let differingRows = 0;
  let firstDifferingExpert: ExpertSignalDiff | undefined;

  for (const key of keys) {
    const left = leftMap.get(key);
    const right = rightMap.get(key);

    if (stableStringify(left) === stableStringify(right)) {
      continue;
    }

    differingRows += 1;

    if (!firstDifferingExpert) {
      const [layerToken, expertToken] = key.split(':');
      const layer = Number(layerToken);
      const expert = Number(expertToken);
      firstDifferingExpert = {
        layer,
        expert,
        ...(left ? { left } : {}),
        ...(right ? { right } : {})
      };
    }
  }

  return {
    differingRows,
    ...(firstDifferingExpert ? { firstDifferingExpert } : {})
  };
}

function toDecisionKeySet(plan: PruningPlan, key: 'pruned' | 'kept'): Set<string> {
  return new Set(plan[key].map((entry) => expertKey(entry.layer, entry.expert)));
}

function decodeDecisionKey(key: string): { layer: number; expert: number } {
  const [layerToken, expertToken] = key.split(':');
  return {
    layer: Number(layerToken),
    expert: Number(expertToken)
  };
}

function sortedDecisionKeys(values: Set<string>): string[] {
  return [...values].sort((left, right) => {
    const leftDecoded = decodeDecisionKey(left);
    const rightDecoded = decodeDecisionKey(right);

    if (leftDecoded.layer !== rightDecoded.layer) {
      return leftDecoded.layer - rightDecoded.layer;
    }

    return leftDecoded.expert - rightDecoded.expert;
  });
}

function diffDecisionSets(
  leftPlan: PruningPlan,
  rightPlan: PruningPlan
): {
  onlyLeftPruned: Array<{ layer: number; expert: number }>;
  onlyRightPruned: Array<{ layer: number; expert: number }>;
  diffByLayer: PruneDiffByLayer[];
} {
  const left = toDecisionKeySet(leftPlan, 'pruned');
  const right = toDecisionKeySet(rightPlan, 'pruned');

  const onlyLeftPruned = sortedDecisionKeys(
    new Set([...left].filter((key) => !right.has(key)))
  ).map((key) => decodeDecisionKey(key));
  const onlyRightPruned = sortedDecisionKeys(
    new Set([...right].filter((key) => !left.has(key)))
  ).map((key) => decodeDecisionKey(key));

  const grouped = new Map<number, PruneDiffByLayer>();

  for (const entry of onlyLeftPruned) {
    const layer = grouped.get(entry.layer) ?? {
      layer: entry.layer,
      onlyLeft: [],
      onlyRight: []
    };
    layer.onlyLeft.push(entry.expert);
    grouped.set(entry.layer, layer);
  }

  for (const entry of onlyRightPruned) {
    const layer = grouped.get(entry.layer) ?? {
      layer: entry.layer,
      onlyLeft: [],
      onlyRight: []
    };
    layer.onlyRight.push(entry.expert);
    grouped.set(entry.layer, layer);
  }

  const diffByLayer = [...grouped.values()]
    .map((entry) => ({
      layer: entry.layer,
      onlyLeft: [...entry.onlyLeft].sort((leftValue, rightValue) => leftValue - rightValue),
      onlyRight: [...entry.onlyRight].sort((leftValue, rightValue) => leftValue - rightValue)
    }))
    .sort((leftValue, rightValue) => leftValue.layer - rightValue.layer);

  return {
    onlyLeftPruned,
    onlyRightPruned,
    diffByLayer
  };
}

function buildRunConfig(
  telemetryPath: string,
  outputDir: string,
  config: ParityHarnessConfig
): RunConfig {
  return {
    modelPath: telemetryPath,
    outputDir,
    ...(typeof config.targetRatio === 'number' ? { targetRatio: config.targetRatio } : {}),
    ...(typeof config.calibrationRounds === 'number'
      ? { calibrationRounds: config.calibrationRounds }
      : {}),
    ...(typeof config.minExpertsPerLayer === 'number'
      ? { minExpertsPerLayer: config.minExpertsPerLayer }
      : {}),
    ...(config.pruneMethod ? { pruneMethod: config.pruneMethod } : {}),
    ...(typeof config.nExpertsToPrunePerLayer === 'number'
      ? { nExpertsToPrunePerLayer: config.nExpertsToPrunePerLayer }
      : {}),
    ...(config.preserveSuperExperts ? { preserveSuperExperts: true } : {}),
    ...(config.preserveOutliers ? { preserveOutliers: true } : {}),
    ...(config.allowLegacySaliency === false ? { allowLegacySaliency: false } : {})
  };
}

function buildMarkdownReport(report: ParityReport): string {
  const lines = [
    '# REAP exact parity report',
    '',
    `- Left telemetry: \`${report.config.leftModelPath}\``,
    `- Right telemetry: \`${report.config.rightModelPath}\``,
    `- Prune method: \`${report.config.pruneMethod}\``,
    '',
    '## Telemetry parity',
    '',
    `- Model name equal: **${report.telemetry.modelNameEqual}**`,
    `- Metadata equal: **${report.telemetry.metadataEqual}**`,
    `- Expert count equal: **${report.telemetry.expertCountEqual}**`,
    `- Normalized hash equal: **${report.telemetry.normalizedHashEqual}**`,
    `- Differing expert rows: **${report.telemetry.differingExpertRows}**`,
    '',
    '## Pruning parity',
    '',
    `- Pruned exact match: **${report.pruning.prunedExactMatch}**`,
    `- Kept exact match: **${report.pruning.keptExactMatch}**`,
    `- Threshold delta: **${report.pruning.thresholdDelta}**`,
    `- Overall exact match: **${report.overallExactMatch}**`,
    ''
  ];

  if (report.pruning.diffByLayer.length > 0) {
    lines.push('## Per-layer prune diff', '', '| Layer | Only left | Only right |', '|---|---|---|');
    for (const entry of report.pruning.diffByLayer) {
      lines.push(
        `| ${entry.layer} | ${entry.onlyLeft.join(', ') || '-'} | ${entry.onlyRight.join(', ') || '-'} |`
      );
    }
    lines.push('');
  }

  if (report.telemetry.firstDifferingExpert) {
    lines.push(
      '## First differing expert row',
      '',
      '```json',
      JSON.stringify(report.telemetry.firstDifferingExpert, null, 2),
      '```',
      ''
    );
  }

  return `${lines.join('\n')}\n`;
}

export async function runParityHarness(config: ParityHarnessConfig): Promise<ParityReport> {
  const outputDir = path.resolve(config.outputDir);
  const leftModelPath = path.resolve(config.leftModelPath);
  const rightModelPath = path.resolve(config.rightModelPath);

  await ensureSecureDir(outputDir);

  const components = createDefaultReapMlxComponents();
  const leftTelemetryRaw = await components.loadTelemetry(leftModelPath);
  const rightTelemetryRaw = await components.loadTelemetry(rightModelPath);
  const leftTelemetry = normalizeTelemetry(components.validateTelemetry(leftTelemetryRaw));
  const rightTelemetry = normalizeTelemetry(components.validateTelemetry(rightTelemetryRaw));

  const telemetryDiff = diffExperts(leftTelemetry.experts, rightTelemetry.experts);
  const leftHash = hashNormalized(leftTelemetry);
  const rightHash = hashNormalized(rightTelemetry);

  const leftOutputDir = resolveSafePath(outputDir, 'left');
  const rightOutputDir = resolveSafePath(outputDir, 'right');
  await ensureSecureDir(leftOutputDir);
  await ensureSecureDir(rightOutputDir);

  const leftPlan = await runReapMlx(buildRunConfig(leftModelPath, leftOutputDir, config));
  const rightPlan = await runReapMlx(buildRunConfig(rightModelPath, rightOutputDir, config));

  const leftPruned = toDecisionKeySet(leftPlan, 'pruned');
  const rightPruned = toDecisionKeySet(rightPlan, 'pruned');
  const leftKept = toDecisionKeySet(leftPlan, 'kept');
  const rightKept = toDecisionKeySet(rightPlan, 'kept');

  const pruningDiff = diffDecisionSets(leftPlan, rightPlan);
  const thresholdDelta = Math.abs(leftPlan.threshold - rightPlan.threshold);
  const telemetryMatches = leftHash === rightHash;
  const pruneSetsEqual = stableStringify(sortedDecisionKeys(leftPruned)) === stableStringify(sortedDecisionKeys(rightPruned));
  const keptSetsEqual = stableStringify(sortedDecisionKeys(leftKept)) === stableStringify(sortedDecisionKeys(rightKept));

  const reportJsonPath = resolveSafePath(outputDir, 'parity-report.json');
  const reportMarkdownPath = resolveSafePath(outputDir, 'parity-report.md');
  const leftPlanPath = resolveSafePath(leftOutputDir, 'pruning-plan.json');
  const rightPlanPath = resolveSafePath(rightOutputDir, 'pruning-plan.json');

  const report: ParityReport = {
    version: '1.0',
    createdAt: new Date().toISOString(),
    config: {
      leftModelPath,
      rightModelPath,
      outputDir,
      ...(typeof config.targetRatio === 'number' ? { targetRatio: config.targetRatio } : {}),
      ...(typeof config.calibrationRounds === 'number'
        ? { calibrationRounds: config.calibrationRounds }
        : {}),
      ...(typeof config.minExpertsPerLayer === 'number'
        ? { minExpertsPerLayer: config.minExpertsPerLayer }
        : {}),
      pruneMethod: config.pruneMethod ?? 'reap',
      ...(typeof config.nExpertsToPrunePerLayer === 'number'
        ? { nExpertsToPrunePerLayer: config.nExpertsToPrunePerLayer }
        : {}),
      preserveSuperExperts: config.preserveSuperExperts ?? false,
      preserveOutliers: config.preserveOutliers ?? false,
      allowLegacySaliency: config.allowLegacySaliency ?? true,
      requireIdenticalTelemetry: config.requireIdenticalTelemetry ?? false
    },
    leftTelemetry: {
      modelName: leftTelemetry.modelName,
      expertCount: leftTelemetry.experts.length,
      ...(leftTelemetry.metadata ? { metadata: leftTelemetry.metadata } : {}),
      normalizedHash: leftHash
    },
    rightTelemetry: {
      modelName: rightTelemetry.modelName,
      expertCount: rightTelemetry.experts.length,
      ...(rightTelemetry.metadata ? { metadata: rightTelemetry.metadata } : {}),
      normalizedHash: rightHash
    },
    telemetry: {
      modelNameEqual: leftTelemetry.modelName === rightTelemetry.modelName,
      metadataEqual:
        stableStringify(leftTelemetry.metadata ?? {}) === stableStringify(rightTelemetry.metadata ?? {}),
      expertCountEqual: leftTelemetry.experts.length === rightTelemetry.experts.length,
      normalizedHashEqual: telemetryMatches,
      differingExpertRows: telemetryDiff.differingRows,
      ...(telemetryDiff.firstDifferingExpert
        ? { firstDifferingExpert: telemetryDiff.firstDifferingExpert }
        : {})
    },
    pruning: {
      thresholdDelta,
      thresholdEqual: thresholdDelta <= 1e-12,
      prunedExactMatch: pruneSetsEqual,
      keptExactMatch: keptSetsEqual,
      onlyLeftPruned: pruningDiff.onlyLeftPruned,
      onlyRightPruned: pruningDiff.onlyRightPruned,
      diffByLayer: pruningDiff.diffByLayer
    },
    overallExactMatch:
      telemetryMatches &&
      pruneSetsEqual &&
      keptSetsEqual &&
      thresholdDelta <= 1e-12,
    artifacts: {
      leftPlanPath,
      rightPlanPath,
      reportJsonPath,
      reportMarkdownPath
    }
  };

  await writeJsonAtomicSafe(reportJsonPath, report);
  await writeTextAtomicSafe(reportMarkdownPath, buildMarkdownReport(report));

  return report;
}
