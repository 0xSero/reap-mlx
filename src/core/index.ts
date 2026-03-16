export { ObservationEngine } from './ObservationEngine.js';
export { runReapMlx } from './reapMlx.js';
export { runParityHarness, type ParityHarnessConfig, type ParityReport } from './parityHarness.js';
export { collectTelemetryWithMlx, type MlxCollectConfig } from './mlxCollector.js';
export {
  probeMlxModelCoherence,
  type MlxCoherenceProbeConfig,
  type MlxCoherenceProbeResult
} from './mlxProbe.js';
export {
  applyPruningPlanToMlxModel,
  deriveOutputPlanPath,
  type MlxApplyPlanConfig,
  type MlxApplyPlanResult
} from './mlxPrune.js';
export {
  createDefaultReapMlxComponents,
  type ReapMlxPipelineComponents,
  type SaliencyScoreResult,
  type ScoredExpert,
  type PruningSelection
} from './reapComponents.js';
export { summarizeObservationLog } from './observe.js';
export type {
  ExpertDecision,
  ExpertSignal,
  ModelTelemetry,
  ObservationEvent,
  ObservationSummary,
  PruneMethod,
  PruningPlan,
  RunConfig,
  SaliencySource
} from './types.js';
