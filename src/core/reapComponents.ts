import {
  assertFiniteNumber,
  assertInteger,
  readJsonFileSafe,
  writeJsonAtomicSafe
} from './security.js';
import type {
  DecisionReason,
  ExpertDecision,
  ExpertSignal,
  ModelTelemetry,
  PruningPlan,
  SaliencySource
} from './types.js';

const MAX_EXPERTS = 200_000;

export interface ScoredExpert extends ExpertSignal {
  key: string;
  signal: number;
  rank: number;
  saliencySource: SaliencySource;
  activeTokenCount: number;
}

export interface SaliencyScoreResult {
  scoredExperts: ScoredExpert[];
  legacyFallbackUsed: boolean;
}

export interface PruningSelection {
  requestedPruneCount: number;
  targetPruneCount: number;
  threshold: number;
  blockedByLayerSafety: number;
  blockedKeys: Set<string>;
  prunedExperts: ScoredExpert[];
  keptExperts: ScoredExpert[];
}

export interface ReapMlxPipelineComponents {
  loadTelemetry: (modelPath: string) => Promise<unknown>;
  validateTelemetry: (input: unknown) => ModelTelemetry;
  scoreSaliency: (
    experts: ExpertSignal[],
    options: {
      allowLegacySaliency: boolean;
    }
  ) => SaliencyScoreResult;
  selectPruning: (
    rankedExperts: ScoredExpert[],
    options: {
      targetRatio: number;
      minExpertsPerLayer: number;
    }
  ) => PruningSelection;
  buildDecisions: (
    rankedExperts: ScoredExpert[],
    selection: PruningSelection
  ) => {
    pruned: ExpertDecision[];
    kept: ExpertDecision[];
  };
  writePlan: (planPath: string, plan: PruningPlan) => Promise<void>;
}

function optionalFinite(
  value: unknown,
  label: string,
  min?: number,
  max?: number
): number | undefined {
  if (typeof value === 'undefined') {
    return undefined;
  }

  return assertFiniteNumber(value, label, min, max);
}

function optionalInteger(
  value: unknown,
  label: string,
  min?: number,
  max?: number
): number | undefined {
  if (typeof value === 'undefined') {
    return undefined;
  }

  return assertInteger(value, label, min, max);
}

function validateExpertSignal(input: unknown, index: number): ExpertSignal {
  if (!input || typeof input !== 'object') {
    throw new Error(`Invalid expert entry at index ${index}`);
  }

  const candidate = input as Record<string, unknown>;

  const layer = assertInteger(candidate.layer, `experts[${index}].layer`, 0, 10_000);
  const expert = assertInteger(candidate.expert, `experts[${index}].expert`, 0, 1_000_000);

  const activationScore = optionalFinite(
    candidate.activationScore,
    `experts[${index}].activationScore`,
    0,
    1_000_000
  );
  const tokenCount = optionalInteger(
    candidate.tokenCount,
    `experts[${index}].tokenCount`,
    0,
    100_000_000
  );
  const activeTokenCount = optionalInteger(
    candidate.activeTokenCount,
    `experts[${index}].activeTokenCount`,
    0,
    100_000_000
  );
  const averageGateValue = optionalFinite(
    candidate.averageGateValue,
    `experts[${index}].averageGateValue`,
    0,
    1
  );
  const averageActivationNorm = optionalFinite(
    candidate.averageActivationNorm,
    `experts[${index}].averageActivationNorm`,
    0,
    1_000_000
  );
  const gateValueSum = optionalFinite(
    candidate.gateValueSum,
    `experts[${index}].gateValueSum`,
    0,
    1_000_000_000
  );
  const activationNormSum = optionalFinite(
    candidate.activationNormSum,
    `experts[${index}].activationNormSum`,
    0,
    1_000_000_000
  );
  const weightedActivationNormSum = optionalFinite(
    candidate.weightedActivationNormSum,
    `experts[${index}].weightedActivationNormSum`,
    0,
    1_000_000_000
  );

  const hasReapSignal =
    typeof weightedActivationNormSum === 'number' ||
    (typeof averageGateValue === 'number' && typeof averageActivationNorm === 'number') ||
    (typeof gateValueSum === 'number' &&
      typeof activationNormSum === 'number' &&
      (typeof activeTokenCount === 'number' || typeof tokenCount === 'number'));

  const hasLegacySignal = typeof activationScore === 'number';

  if (!hasReapSignal && !hasLegacySignal) {
    throw new Error(
      `experts[${index}] must include REAP saliency fields or activationScore fallback`
    );
  }

  return {
    layer,
    expert,
    ...(typeof activationScore === 'number' ? { activationScore } : {}),
    ...(typeof tokenCount === 'number' ? { tokenCount } : {}),
    ...(typeof activeTokenCount === 'number' ? { activeTokenCount } : {}),
    ...(typeof averageGateValue === 'number' ? { averageGateValue } : {}),
    ...(typeof averageActivationNorm === 'number' ? { averageActivationNorm } : {}),
    ...(typeof gateValueSum === 'number' ? { gateValueSum } : {}),
    ...(typeof activationNormSum === 'number' ? { activationNormSum } : {}),
    ...(typeof weightedActivationNormSum === 'number' ? { weightedActivationNormSum } : {})
  };
}

type MetadataValue = string | number | boolean | Record<string, string | number>;

function normalizeMetadata(
  value: unknown
): Record<string, MetadataValue> | undefined {
  if (typeof value === 'undefined') {
    return undefined;
  }

  if (!value || typeof value !== 'object') {
    throw new Error('metadata must be an object when provided');
  }

  const metadata = value as Record<string, unknown>;
  const normalized: Record<string, MetadataValue> = {};

  for (const [key, candidate] of Object.entries(metadata)) {
    if (
      typeof candidate === 'string' ||
      typeof candidate === 'number' ||
      typeof candidate === 'boolean'
    ) {
      normalized[key] = candidate;
      continue;
    }

    if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
      const obj = candidate as Record<string, unknown>;
      const mapped: Record<string, string | number> = {};

      for (const [subKey, subValue] of Object.entries(obj)) {
        if (typeof subValue === 'string' || typeof subValue === 'number') {
          mapped[subKey] = subValue;
          continue;
        }

        throw new Error(
          `metadata.${key}.${subKey} must be string or number`
        );
      }

      normalized[key] = mapped;
      continue;
    }

    throw new Error(`metadata.${key} must be scalar or flat object`);
  }

  return normalized;
}

function validateModelTelemetry(input: unknown): ModelTelemetry {
  if (!input || typeof input !== 'object') {
    throw new Error('Model telemetry must be a JSON object');
  }

  const candidate = input as Record<string, unknown>;

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

  const metadata = normalizeMetadata(candidate.metadata);

  return {
    modelName: candidate.modelName,
    experts,
    ...(metadata ? { metadata } : {})
  };
}

function resolveActiveTokenCount(expert: ExpertSignal): number {
  if (typeof expert.activeTokenCount === 'number') {
    return expert.activeTokenCount;
  }

  if (typeof expert.tokenCount === 'number') {
    return expert.tokenCount;
  }

  return 0;
}

function roundSignal(value: number): number {
  return Number(value.toFixed(8));
}

function scoreSaliency(
  experts: ExpertSignal[],
  options: {
    allowLegacySaliency: boolean;
  }
): SaliencyScoreResult {
  const scoredExperts = [] as ScoredExpert[];
  let legacyFallbackUsed = false;

  for (const expert of experts) {
    const activeTokenCount = resolveActiveTokenCount(expert);

    let signal: number | undefined;
    let saliencySource: SaliencySource | undefined;

    if (
      typeof expert.weightedActivationNormSum === 'number' &&
      activeTokenCount > 0
    ) {
      signal = expert.weightedActivationNormSum / activeTokenCount;
      saliencySource = 'weighted_activation_sum';
    } else if (
      typeof expert.averageGateValue === 'number' &&
      typeof expert.averageActivationNorm === 'number'
    ) {
      signal = expert.averageGateValue * expert.averageActivationNorm;
      saliencySource = 'mean_gate_x_norm';
    } else if (
      typeof expert.gateValueSum === 'number' &&
      typeof expert.activationNormSum === 'number' &&
      activeTokenCount > 0
    ) {
      const meanGate = expert.gateValueSum / activeTokenCount;
      const meanNorm = expert.activationNormSum / activeTokenCount;
      signal = meanGate * meanNorm;
      saliencySource = 'mean_gate_x_norm_from_sums';
    } else if (
      options.allowLegacySaliency &&
      typeof expert.activationScore === 'number'
    ) {
      const tokenWeight = Math.log2((activeTokenCount || 1) + 2);
      signal = expert.activationScore * tokenWeight;
      saliencySource = 'legacy_activation_score';
      legacyFallbackUsed = true;
    }

    if (typeof signal !== 'number' || typeof saliencySource === 'undefined') {
      throw new Error(
        `Missing REAP saliency fields for layer=${expert.layer} expert=${expert.expert}`
      );
    }

    scoredExperts.push({
      layer: expert.layer,
      expert: expert.expert,
      ...(typeof expert.activationScore === 'number'
        ? { activationScore: expert.activationScore }
        : {}),
      ...(typeof expert.tokenCount === 'number' ? { tokenCount: expert.tokenCount } : {}),
      ...(typeof expert.activeTokenCount === 'number'
        ? { activeTokenCount: expert.activeTokenCount }
        : {}),
      ...(typeof expert.averageGateValue === 'number'
        ? { averageGateValue: expert.averageGateValue }
        : {}),
      ...(typeof expert.averageActivationNorm === 'number'
        ? { averageActivationNorm: expert.averageActivationNorm }
        : {}),
      ...(typeof expert.gateValueSum === 'number' ? { gateValueSum: expert.gateValueSum } : {}),
      ...(typeof expert.activationNormSum === 'number'
        ? { activationNormSum: expert.activationNormSum }
        : {}),
      ...(typeof expert.weightedActivationNormSum === 'number'
        ? { weightedActivationNormSum: expert.weightedActivationNormSum }
        : {}),
      key: `${expert.layer}:${expert.expert}`,
      signal: roundSignal(signal),
      rank: 0,
      saliencySource,
      activeTokenCount
    });
  }

  scoredExperts.sort((left, right) => {
    if (left.signal !== right.signal) {
      return left.signal - right.signal;
    }

    if (left.layer !== right.layer) {
      return left.layer - right.layer;
    }

    return left.expert - right.expert;
  });

  scoredExperts.forEach((expert, index) => {
    expert.rank = index + 1;
  });

  return {
    scoredExperts,
    legacyFallbackUsed
  };
}

function selectPruning(
  rankedExperts: ScoredExpert[],
  options: {
    targetRatio: number;
    minExpertsPerLayer: number;
  }
): PruningSelection {
  const expertsByLayer = new Map<number, ScoredExpert[]>();

  for (const expert of rankedExperts) {
    const layerExperts = expertsByLayer.get(expert.layer);
    if (layerExperts) {
      layerExperts.push(expert);
    } else {
      expertsByLayer.set(expert.layer, [expert]);
    }
  }

  const prunedExperts = [] as ScoredExpert[];
  const blockedKeys = new Set<string>();

  let requestedPruneCount = 0;
  let targetPruneCount = 0;

  for (const layerExperts of expertsByLayer.values()) {
    const sortedLayerExperts = [...layerExperts].sort((left, right) => {
      if (left.signal !== right.signal) {
        return left.signal - right.signal;
      }

      return left.expert - right.expert;
    });

    const layerRequested = Math.floor(sortedLayerExperts.length * options.targetRatio);
    const layerMaxPrunable = Math.max(
      0,
      sortedLayerExperts.length - options.minExpertsPerLayer
    );
    const layerTarget = Math.min(layerRequested, layerMaxPrunable);

    requestedPruneCount += layerRequested;
    targetPruneCount += layerTarget;

    for (const expert of sortedLayerExperts.slice(0, layerTarget)) {
      prunedExperts.push(expert);
    }

    if (layerRequested > layerTarget) {
      const blocked = sortedLayerExperts.slice(layerTarget, layerRequested);
      for (const expert of blocked) {
        blockedKeys.add(expert.key);
      }
    }
  }

  const prunedKeys = new Set(prunedExperts.map((expert) => expert.key));
  const keptExperts = rankedExperts.filter((expert) => !prunedKeys.has(expert.key));
  const threshold = prunedExperts[prunedExperts.length - 1]?.signal ?? 0;

  return {
    requestedPruneCount,
    targetPruneCount,
    threshold,
    blockedByLayerSafety: blockedKeys.size,
    blockedKeys,
    prunedExperts,
    keptExperts
  };
}

function toDecision(
  expert: ScoredExpert,
  reason: DecisionReason
): ExpertDecision {
  return {
    layer: expert.layer,
    expert: expert.expert,
    ...(typeof expert.activationScore === 'number'
      ? { activationScore: expert.activationScore }
      : {}),
    ...(typeof expert.tokenCount === 'number' ? { tokenCount: expert.tokenCount } : {}),
    ...(typeof expert.activeTokenCount === 'number'
      ? { activeTokenCount: expert.activeTokenCount }
      : {}),
    ...(typeof expert.averageGateValue === 'number'
      ? { averageGateValue: expert.averageGateValue }
      : {}),
    ...(typeof expert.averageActivationNorm === 'number'
      ? { averageActivationNorm: expert.averageActivationNorm }
      : {}),
    ...(typeof expert.gateValueSum === 'number' ? { gateValueSum: expert.gateValueSum } : {}),
    ...(typeof expert.activationNormSum === 'number'
      ? { activationNormSum: expert.activationNormSum }
      : {}),
    ...(typeof expert.weightedActivationNormSum === 'number'
      ? { weightedActivationNormSum: expert.weightedActivationNormSum }
      : {}),
    signal: expert.signal,
    rank: expert.rank,
    reason,
    saliencySource: expert.saliencySource,
    activeTokenCount: expert.activeTokenCount
  };
}

function buildDecisions(
  rankedExperts: ScoredExpert[],
  selection: PruningSelection
): {
  pruned: ExpertDecision[];
  kept: ExpertDecision[];
} {
  const prunedKeys = new Set(selection.prunedExperts.map((expert) => expert.key));

  const pruned = [] as ExpertDecision[];
  const kept = [] as ExpertDecision[];

  for (const expert of rankedExperts) {
    if (prunedKeys.has(expert.key)) {
      pruned.push(toDecision(expert, 'low_signal_pruned'));
      continue;
    }

    const reason: DecisionReason = selection.blockedKeys.has(expert.key)
      ? 'retained_for_layer_safety'
      : 'high_signal_retained';

    kept.push(toDecision(expert, reason));
  }

  return {
    pruned,
    kept
  };
}

export function createDefaultReapMlxComponents(): ReapMlxPipelineComponents {
  return {
    loadTelemetry: async (modelPath: string): Promise<unknown> => readJsonFileSafe<unknown>(modelPath),
    validateTelemetry: validateModelTelemetry,
    scoreSaliency,
    selectPruning,
    buildDecisions,
    writePlan: async (planPath: string, plan: PruningPlan): Promise<void> =>
      writeJsonAtomicSafe(planPath, plan)
  };
}
