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
  PruneMethod,
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
      pruneMethod: PruneMethod;
    }
  ) => SaliencyScoreResult;
  selectPruning: (
    rankedExperts: ScoredExpert[],
    options: {
      targetRatio: number;
      minExpertsPerLayer: number;
      nExpertsToPrunePerLayer?: number;
      preserveSuperExperts: boolean;
      preserveOutliers: boolean;
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

function pickFirstFinite(...values: Array<number | undefined>): number | undefined {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }

  return undefined;
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
  const frequency = optionalFinite(
    candidate.frequency,
    `experts[${index}].frequency`,
    0,
    1_000_000_000
  );
  const eanSum = optionalFinite(
    candidate.eanSum,
    `experts[${index}].eanSum`,
    0,
    1_000_000_000
  );
  const eanMean = optionalFinite(
    candidate.eanMean,
    `experts[${index}].eanMean`,
    0,
    1_000_000_000
  );
  const weightedEanSum = optionalFinite(
    candidate.weightedEanSum,
    `experts[${index}].weightedEanSum`,
    0,
    1_000_000_000
  );
  const reap = optionalFinite(
    candidate.reap,
    `experts[${index}].reap`,
    0,
    1_000_000_000
  );
  const maxActivation = optionalFinite(
    candidate.maxActivation,
    `experts[${index}].maxActivation`,
    0,
    1_000_000_000
  );
  const maxActivationNorm = optionalFinite(
    candidate.maxActivationNorm,
    `experts[${index}].maxActivationNorm`,
    0,
    1_000_000_000
  );

  const hasReapSignal =
    typeof weightedActivationNormSum === 'number' ||
    (typeof averageGateValue === 'number' && typeof averageActivationNorm === 'number') ||
    (typeof gateValueSum === 'number' &&
      typeof activationNormSum === 'number' &&
      (typeof activeTokenCount === 'number' || typeof tokenCount === 'number'));

  const hasParitySignal =
    typeof frequency === 'number' ||
    typeof eanSum === 'number' ||
    typeof eanMean === 'number' ||
    typeof weightedEanSum === 'number' ||
    typeof reap === 'number';

  const hasLegacySignal = typeof activationScore === 'number';

  if (!hasReapSignal && !hasParitySignal && !hasLegacySignal) {
    throw new Error(
      `experts[${index}] must include pruning saliency fields or activationScore fallback`
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
    ...(typeof weightedActivationNormSum === 'number' ? { weightedActivationNormSum } : {}),
    ...(typeof frequency === 'number' ? { frequency } : {}),
    ...(typeof eanSum === 'number' ? { eanSum } : {}),
    ...(typeof eanMean === 'number' ? { eanMean } : {}),
    ...(typeof weightedEanSum === 'number' ? { weightedEanSum } : {}),
    ...(typeof reap === 'number' ? { reap } : {}),
    ...(typeof maxActivation === 'number' ? { maxActivation } : {}),
    ...(typeof maxActivationNorm === 'number' ? { maxActivationNorm } : {})
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
    pruneMethod: PruneMethod;
  }
): SaliencyScoreResult {
  const scoredExperts = [] as ScoredExpert[];
  let legacyFallbackUsed = false;

  for (const expert of experts) {
    const activeTokenCount = resolveActiveTokenCount(expert);

    let signal: number | undefined;
    let saliencySource: SaliencySource | undefined;

    if (options.pruneMethod === 'frequency') {
      signal = pickFirstFinite(expert.frequency, expert.activeTokenCount, expert.tokenCount);
      saliencySource = 'frequency';
    } else if (options.pruneMethod === 'ean_sum') {
      signal = pickFirstFinite(expert.eanSum, expert.activationNormSum);
      saliencySource = 'ean_sum';
    } else if (options.pruneMethod === 'ean_mean') {
      if (typeof expert.eanMean === 'number') {
        signal = expert.eanMean;
      } else if (typeof expert.averageActivationNorm === 'number') {
        signal = expert.averageActivationNorm;
      } else if (typeof expert.activationNormSum === 'number' && activeTokenCount > 0) {
        signal = expert.activationNormSum / activeTokenCount;
      }
      saliencySource = 'ean_mean';
    } else if (options.pruneMethod === 'weighted_ean_sum') {
      signal = pickFirstFinite(expert.weightedEanSum, expert.weightedActivationNormSum);
      saliencySource = 'weighted_ean_sum';
    } else {
      if (typeof expert.reap === 'number') {
        signal = expert.reap;
        saliencySource = 'reap';
      } else if (
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
      }
    }

    if (
      options.allowLegacySaliency &&
      typeof signal !== 'number' &&
      typeof expert.activationScore === 'number'
    ) {
      const tokenWeight = Math.log2((activeTokenCount || 1) + 2);
      signal = expert.activationScore * tokenWeight;
      saliencySource = 'legacy_activation_score';
      legacyFallbackUsed = true;
    }

    if (typeof signal !== 'number' || typeof saliencySource === 'undefined') {
      throw new Error(
        `Missing saliency fields for pruneMethod=${options.pruneMethod} layer=${expert.layer} expert=${expert.expert}`
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
      ...(typeof expert.frequency === 'number' ? { frequency: expert.frequency } : {}),
      ...(typeof expert.eanSum === 'number' ? { eanSum: expert.eanSum } : {}),
      ...(typeof expert.eanMean === 'number' ? { eanMean: expert.eanMean } : {}),
      ...(typeof expert.weightedEanSum === 'number'
        ? { weightedEanSum: expert.weightedEanSum }
        : {}),
      ...(typeof expert.reap === 'number' ? { reap: expert.reap } : {}),
      ...(typeof expert.maxActivation === 'number' ? { maxActivation: expert.maxActivation } : {}),
      ...(typeof expert.maxActivationNorm === 'number'
        ? { maxActivationNorm: expert.maxActivationNorm }
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

function buildPreservedExpertKeySet(
  rankedExperts: ScoredExpert[],
  options: {
    preserveSuperExperts: boolean;
    preserveOutliers: boolean;
  }
): Set<string> {
  if (!options.preserveSuperExperts && !options.preserveOutliers) {
    return new Set<string>();
  }

  if (options.preserveSuperExperts && options.preserveOutliers) {
    throw new Error('preserveSuperExperts and preserveOutliers cannot both be true');
  }

  const expertsByLayer = new Map<number, ScoredExpert[]>();
  for (const expert of rankedExperts) {
    const layerExperts = expertsByLayer.get(expert.layer);
    if (layerExperts) {
      layerExperts.push(expert);
      continue;
    }

    expertsByLayer.set(expert.layer, [expert]);
  }

  const allLayers = [...expertsByLayer.keys()].sort((left, right) => left - right);
  if (allLayers.length === 0) {
    return new Set<string>();
  }

  const layerCutoff = options.preserveOutliers
    ? allLayers.length
    : Math.floor(allLayers.length * 0.75);

  const eligibleLayers = new Set(allLayers.slice(0, layerCutoff));

  const maxActivations = [] as number[];

  for (const expert of rankedExperts) {
    const maxActivation = pickFirstFinite(expert.maxActivation, expert.maxActivationNorm);
    if (typeof maxActivation === 'number') {
      maxActivations.push(maxActivation);
    }
  }

  if (maxActivations.length === 0) {
    return new Set<string>();
  }

  maxActivations.sort((left, right) => left - right);

  const quantilePosition = ((maxActivations.length - 1) * 99.5) / 100;
  const quantileFloor = Math.floor(quantilePosition);
  const quantileCeil = Math.ceil(quantilePosition);
  const quantileFloorValue =
    maxActivations[quantileFloor] ?? maxActivations[maxActivations.length - 1] ?? 0;
  const quantileCeilValue =
    maxActivations[quantileCeil] ?? maxActivations[maxActivations.length - 1] ?? 0;
  const quantileWeight = quantilePosition - quantileFloor;
  const percentileThreshold =
    quantileFloorValue + (quantileCeilValue - quantileFloorValue) * quantileWeight;
  const maxValue = maxActivations[maxActivations.length - 1] ?? 0;
  const absoluteThreshold = maxValue / 10;
  const finalThreshold = Math.max(percentileThreshold, absoluteThreshold);

  const preservedKeys = new Set<string>();

  for (const expert of rankedExperts) {
    if (!eligibleLayers.has(expert.layer)) {
      continue;
    }

    const maxActivation = pickFirstFinite(expert.maxActivation, expert.maxActivationNorm);
    if (typeof maxActivation !== 'number') {
      continue;
    }

    if (maxActivation > finalThreshold) {
      preservedKeys.add(expert.key);
    }
  }

  return preservedKeys;
}

function selectPruning(
  rankedExperts: ScoredExpert[],
  options: {
    targetRatio: number;
    minExpertsPerLayer: number;
    nExpertsToPrunePerLayer?: number;
    preserveSuperExperts: boolean;
    preserveOutliers: boolean;
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

  const preservedKeys = buildPreservedExpertKeySet(rankedExperts, {
    preserveSuperExperts: options.preserveSuperExperts,
    preserveOutliers: options.preserveOutliers
  });

  const prunedExperts = [] as ScoredExpert[];
  const blockedKeys = new Set<string>();

  let requestedPruneCount = 0;
  let targetPruneCount = 0;

  for (const layerExperts of expertsByLayer.values()) {
    const sortedLayerExperts = [...layerExperts].sort((left, right) => {
      const leftSignal = preservedKeys.has(left.key) ? Number.POSITIVE_INFINITY : left.signal;
      const rightSignal = preservedKeys.has(right.key) ? Number.POSITIVE_INFINITY : right.signal;

      if (leftSignal !== rightSignal) {
        return leftSignal - rightSignal;
      }

      return left.expert - right.expert;
    });

    const layerRequested =
      typeof options.nExpertsToPrunePerLayer === 'number'
        ? Math.min(sortedLayerExperts.length, options.nExpertsToPrunePerLayer)
        : Math.floor(sortedLayerExperts.length * options.targetRatio);

    const layerMaxPrunableBySafety = Math.max(
      0,
      sortedLayerExperts.length - options.minExpertsPerLayer
    );
    const layerMaxPrunableByPreserve = sortedLayerExperts.filter(
      (expert) => !preservedKeys.has(expert.key)
    ).length;

    const layerTarget =
      typeof options.nExpertsToPrunePerLayer === 'number'
        ? Math.min(layerRequested, layerMaxPrunableByPreserve)
        : Math.min(
            layerRequested,
            layerMaxPrunableBySafety,
            layerMaxPrunableByPreserve
          );

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
    ...(typeof expert.frequency === 'number' ? { frequency: expert.frequency } : {}),
    ...(typeof expert.eanSum === 'number' ? { eanSum: expert.eanSum } : {}),
    ...(typeof expert.eanMean === 'number' ? { eanMean: expert.eanMean } : {}),
    ...(typeof expert.weightedEanSum === 'number'
      ? { weightedEanSum: expert.weightedEanSum }
      : {}),
    ...(typeof expert.reap === 'number' ? { reap: expert.reap } : {}),
    ...(typeof expert.maxActivation === 'number' ? { maxActivation: expert.maxActivation } : {}),
    ...(typeof expert.maxActivationNorm === 'number'
      ? { maxActivationNorm: expert.maxActivationNorm }
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
