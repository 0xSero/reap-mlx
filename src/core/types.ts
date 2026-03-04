export type ObservationStage =
  | 'bootstrap'
  | 'load_model'
  | 'validate'
  | 'score_experts'
  | 'plan_pruning'
  | 'write_output'
  | 'complete'
  | 'observe';

export type ObservationLevel = 'info' | 'warn' | 'error';

export interface ExpertSignal {
  layer: number;
  expert: number;
  activationScore?: number;
  tokenCount?: number;
  activeTokenCount?: number;
  averageGateValue?: number;
  averageActivationNorm?: number;
  gateValueSum?: number;
  activationNormSum?: number;
  weightedActivationNormSum?: number;
}

export type TelemetryMetadataValue =
  | string
  | number
  | boolean
  | Record<string, string | number>;

export interface ModelTelemetry {
  modelName: string;
  experts: ExpertSignal[];
  metadata?: Record<string, TelemetryMetadataValue>;
}

export type DecisionReason =
  | 'low_signal_pruned'
  | 'retained_for_layer_safety'
  | 'high_signal_retained';

export type SaliencySource =
  | 'weighted_activation_sum'
  | 'mean_gate_x_norm'
  | 'mean_gate_x_norm_from_sums'
  | 'legacy_activation_score';

export interface ExpertDecision extends ExpertSignal {
  signal: number;
  rank: number;
  reason: DecisionReason;
  saliencySource: SaliencySource;
  activeTokenCount: number;
}

export interface PruningPlan {
  version: '1.0';
  createdAt: string;
  jobId: string;
  modelName: string;
  targetRatio: number;
  achievedRatio: number;
  calibrationRounds: number;
  saliencyMethod: 'reap';
  minExpertsPerLayer: number;
  legacyFallbackUsed: boolean;
  threshold: number;
  stats: {
    totalExperts: number;
    prunedExperts: number;
    keptExperts: number;
    blockedByLayerSafety: number;
  };
  pruned: ExpertDecision[];
  kept: ExpertDecision[];
}

export interface RunConfig {
  modelPath: string;
  outputDir: string;
  targetRatio: number;
  calibrationRounds?: number;
  minExpertsPerLayer?: number;
  allowLegacySaliency?: boolean;
  jobId?: string;
  observationPath?: string;
}

export interface ObservationEvent {
  id: string;
  jobId: string;
  timestamp: string;
  stage: ObservationStage;
  level: ObservationLevel;
  message: string;
  durationMs?: number;
  progressPercent?: number;
  data?: Record<string, unknown>;
}

export interface ObservationSummary {
  jobId?: string;
  totalEvents: number;
  malformedLines: number;
  startedAt?: string;
  endedAt?: string;
  totalDurationMs: number;
  levels: Record<ObservationLevel, number>;
  stageCounts: Partial<Record<ObservationStage, number>>;
  stageDurationsMs: Partial<Record<ObservationStage, number>>;
}
