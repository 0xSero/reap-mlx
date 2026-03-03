import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { ObservationEngine } from './ObservationEngine.js';
import {
  assertFiniteNumber,
  assertInteger,
  ensureSecureDir,
  readJsonFileSafe,
  resolveSafePath,
  writeJsonAtomicSafe
} from './security.js';
import type {
  ExpertDecision,
  ExpertSignal,
  ModelTelemetry,
  PruningPlan,
  RunConfig
} from './types.js';

const MAX_EXPERTS = 200_000;
const DEFAULT_CALIBRATION_ROUNDS = 2;

interface ScoredExpert extends ExpertSignal {
  signal: number;
  rank: number;
  key: string;
}

function validateExpertSignal(input: unknown, index: number): ExpertSignal {
  if (!input || typeof input !== 'object') {
    throw new Error(`Invalid expert entry at index ${index}`);
  }

  const candidate = input as Partial<ExpertSignal>;

  const layer = assertInteger(candidate.layer, `experts[${index}].layer`, 0, 10_000);
  const expert = assertInteger(
    candidate.expert,
    `experts[${index}].expert`,
    0,
    1_000_000
  );
  const activationScore = assertFiniteNumber(
    candidate.activationScore,
    `experts[${index}].activationScore`,
    0,
    1_000_000
  );

  const tokenCount =
    typeof candidate.tokenCount === 'undefined'
      ? undefined
      : assertInteger(
          candidate.tokenCount,
          `experts[${index}].tokenCount`,
          0,
          100_000_000
        );

  return {
    layer,
    expert,
    activationScore,
    ...(typeof tokenCount === 'number' ? { tokenCount } : {})
  };
}

function validateModelTelemetry(input: unknown): ModelTelemetry {
  if (!input || typeof input !== 'object') {
    throw new Error('Model telemetry must be a JSON object');
  }

  const candidate = input as Partial<ModelTelemetry>;

  if (typeof candidate.modelName !== 'string' || candidate.modelName.trim().length === 0) {
    throw new Error('modelName must be a non-empty string');
  }

  if (!Array.isArray(candidate.experts)) {
    throw new Error('experts must be an array');
  }

  if (candidate.experts.length === 0) {
    throw new Error('experts array cannot be empty');
  }

  if (candidate.experts.length > MAX_EXPERTS) {
    throw new Error(`experts array too large (max ${MAX_EXPERTS})`);
  }

  const experts = candidate.experts.map((item, index) => validateExpertSignal(item, index));
  const uniqueKeys = new Set<string>();

  for (const expert of experts) {
    const key = `${expert.layer}:${expert.expert}`;
    if (uniqueKeys.has(key)) {
      throw new Error(`Duplicate expert index detected: ${key}`);
    }
    uniqueKeys.add(key);
  }

  return {
    modelName: candidate.modelName,
    experts,
    ...(candidate.metadata ? { metadata: candidate.metadata } : {})
  };
}

function scoreExpert(expert: ExpertSignal): number {
  const tokenWeight = Math.log2((expert.tokenCount ?? 1) + 2);
  return Number((expert.activationScore * tokenWeight).toFixed(8));
}

function scoreExperts(experts: ExpertSignal[]): ScoredExpert[] {
  const scored = experts
    .map((expert) => ({
      ...expert,
      signal: scoreExpert(expert),
      rank: 0,
      key: `${expert.layer}:${expert.expert}`
    }))
    .sort((left, right) => left.signal - right.signal);

  scored.forEach((expert, index) => {
    expert.rank = index + 1;
  });

  return scored;
}

function selectPrunedExperts(
  rankedExperts: ScoredExpert[],
  targetPrune: number
): { pruned: ScoredExpert[]; blockedByLayerSafety: number } {
  const remainingExpertsPerLayer = new Map<number, number>();

  for (const expert of rankedExperts) {
    remainingExpertsPerLayer.set(
      expert.layer,
      (remainingExpertsPerLayer.get(expert.layer) ?? 0) + 1
    );
  }

  const pruned: ScoredExpert[] = [];
  let blockedByLayerSafety = 0;

  for (const expert of rankedExperts) {
    if (pruned.length >= targetPrune) {
      break;
    }

    const remainingOnLayer = remainingExpertsPerLayer.get(expert.layer);
    if (typeof remainingOnLayer !== 'number') {
      continue;
    }

    if (remainingOnLayer <= 1) {
      blockedByLayerSafety += 1;
      continue;
    }

    remainingExpertsPerLayer.set(expert.layer, remainingOnLayer - 1);
    pruned.push(expert);
  }

  return {
    pruned,
    blockedByLayerSafety
  };
}

function buildDecisionMaps(
  rankedExperts: ScoredExpert[],
  prunedExperts: ScoredExpert[]
): { pruned: ExpertDecision[]; kept: ExpertDecision[]; threshold: number } {
  const prunedKeys = new Set(prunedExperts.map((expert) => expert.key));
  const threshold = prunedExperts[prunedExperts.length - 1]?.signal ?? 0;

  const pruned: ExpertDecision[] = [];
  const kept: ExpertDecision[] = [];

  for (const expert of rankedExperts) {
    if (prunedKeys.has(expert.key)) {
      pruned.push({
        layer: expert.layer,
        expert: expert.expert,
        activationScore: expert.activationScore,
        ...(typeof expert.tokenCount === 'number' ? { tokenCount: expert.tokenCount } : {}),
        signal: expert.signal,
        rank: expert.rank,
        reason: 'low_signal_pruned'
      });
      continue;
    }

    kept.push({
      layer: expert.layer,
      expert: expert.expert,
      activationScore: expert.activationScore,
      ...(typeof expert.tokenCount === 'number' ? { tokenCount: expert.tokenCount } : {}),
      signal: expert.signal,
      rank: expert.rank,
      reason:
        expert.signal <= threshold
          ? 'retained_for_layer_safety'
          : 'high_signal_retained'
    });
  }

  return {
    pruned,
    kept,
    threshold
  };
}

export async function runReapMlx(config: RunConfig): Promise<PruningPlan> {
  const targetRatio = assertFiniteNumber(config.targetRatio, 'targetRatio', 0, 0.95);
  const calibrationRounds = assertInteger(
    config.calibrationRounds ?? DEFAULT_CALIBRATION_ROUNDS,
    'calibrationRounds',
    1,
    25
  );

  const outputDir = path.resolve(config.outputDir);
  const modelPath = path.resolve(config.modelPath);
  await ensureSecureDir(outputDir);

  const observationRelativePath = config.observationPath ?? 'observation.log';
  const observationPath = resolveSafePath(outputDir, observationRelativePath);
  const planPath = resolveSafePath(outputDir, 'pruning-plan.json');

  const observer = new ObservationEngine({
    sinkFilePath: observationPath,
    maxEvents: 5_000,
    jobId: config.jobId ?? randomUUID()
  });

  observer.record('bootstrap', 'reap-mlx run started', {
    data: {
      modelPath,
      outputDir,
      targetRatio,
      calibrationRounds
    }
  });

  const telemetryRaw = await observer.track('load_model', 'Loaded telemetry file', async () =>
    readJsonFileSafe<unknown>(modelPath)
  );

  const telemetry = await observer.track('validate', 'Validated telemetry payload', async () =>
    validateModelTelemetry(telemetryRaw)
  );

  const rankedExperts = await observer.track('score_experts', 'Scored experts for pruning', async () =>
    scoreExperts(telemetry.experts)
  );

  const uniqueLayers = new Set(rankedExperts.map((expert) => expert.layer)).size;
  const maxPrunable = Math.max(0, rankedExperts.length - uniqueLayers);
  const requestedPruneCount = Math.floor(rankedExperts.length * targetRatio);
  const targetPrune = Math.min(requestedPruneCount, maxPrunable);

  let selected = selectPrunedExperts(rankedExperts, targetPrune);

  for (
    let round = 1;
    round < calibrationRounds && selected.pruned.length < targetPrune;
    round += 1
  ) {
    const adjustedTarget = Math.max(0, targetPrune - round);
    selected = selectPrunedExperts(rankedExperts, adjustedTarget);
  }

  const decisions = buildDecisionMaps(rankedExperts, selected.pruned);

  observer.record('plan_pruning', 'Pruning plan computed', {
    data: {
      totalExperts: rankedExperts.length,
      requestedPruneCount,
      effectivePruneCount: decisions.pruned.length,
      targetPrune,
      blockedByLayerSafety: selected.blockedByLayerSafety,
      threshold: decisions.threshold
    },
    progressPercent: 85
  });

  const plan: PruningPlan = {
    version: '1.0',
    createdAt: new Date().toISOString(),
    jobId: observer.getJobId(),
    modelName: telemetry.modelName,
    targetRatio,
    achievedRatio:
      rankedExperts.length === 0
        ? 0
        : Number((decisions.pruned.length / rankedExperts.length).toFixed(6)),
    calibrationRounds,
    threshold: decisions.threshold,
    stats: {
      totalExperts: rankedExperts.length,
      prunedExperts: decisions.pruned.length,
      keptExperts: decisions.kept.length,
      blockedByLayerSafety: selected.blockedByLayerSafety
    },
    pruned: decisions.pruned,
    kept: decisions.kept
  };

  await observer.track('write_output', 'Wrote pruning plan to disk', async () =>
    writeJsonAtomicSafe(planPath, plan)
  );

  observer.record('complete', 'reap-mlx run completed', {
    data: {
      planPath,
      observationPath,
      achievedRatio: plan.achievedRatio
    },
    progressPercent: 100
  });

  await observer.flush();
  return plan;
}
