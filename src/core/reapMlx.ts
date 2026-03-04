import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { ObservationEngine } from './ObservationEngine.js';
import {
  assertFiniteNumber,
  assertInteger,
  ensureSecureDir,
  resolveSafePath
} from './security.js';
import { createDefaultReapMlxComponents } from './reapComponents.js';
import type { PruningPlan, RunConfig } from './types.js';

const DEFAULT_CALIBRATION_ROUNDS = 2;
const DEFAULT_MIN_EXPERTS_PER_LAYER = 1;

export async function runReapMlx(config: RunConfig): Promise<PruningPlan> {
  const targetRatio = assertFiniteNumber(config.targetRatio, 'targetRatio', 0, 0.95);
  const calibrationRounds = assertInteger(
    config.calibrationRounds ?? DEFAULT_CALIBRATION_ROUNDS,
    'calibrationRounds',
    1,
    25
  );
  const minExpertsPerLayer = assertInteger(
    config.minExpertsPerLayer ?? DEFAULT_MIN_EXPERTS_PER_LAYER,
    'minExpertsPerLayer',
    1,
    128
  );

  const allowLegacySaliency = config.allowLegacySaliency ?? true;

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

  const components = createDefaultReapMlxComponents();

  observer.record('bootstrap', 'reap-mlx run started', {
    data: {
      modelPath,
      outputDir,
      targetRatio,
      calibrationRounds,
      minExpertsPerLayer,
      allowLegacySaliency
    }
  });

  const telemetryRaw = await observer.track('load_model', 'Loaded telemetry file', async () =>
    components.loadTelemetry(modelPath)
  );

  const telemetry = await observer.track('validate', 'Validated telemetry payload', async () =>
    components.validateTelemetry(telemetryRaw)
  );

  const saliency = await observer.track('score_experts', 'Computed REAP saliency scores', async () =>
    components.scoreSaliency(telemetry.experts, {
      allowLegacySaliency
    })
  );

  const selection = components.selectPruning(saliency.scoredExperts, {
    targetRatio,
    minExpertsPerLayer
  });

  const decisions = components.buildDecisions(saliency.scoredExperts, selection);

  observer.record('plan_pruning', 'Pruning plan computed', {
    data: {
      saliencyMethod: 'reap',
      legacyFallbackUsed: saliency.legacyFallbackUsed,
      totalExperts: saliency.scoredExperts.length,
      requestedPruneCount: selection.requestedPruneCount,
      effectivePruneCount: decisions.pruned.length,
      targetPruneCount: selection.targetPruneCount,
      minExpertsPerLayer,
      blockedByLayerSafety: selection.blockedByLayerSafety,
      threshold: selection.threshold
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
      saliency.scoredExperts.length === 0
        ? 0
        : Number((decisions.pruned.length / saliency.scoredExperts.length).toFixed(6)),
    calibrationRounds,
    saliencyMethod: 'reap',
    minExpertsPerLayer,
    legacyFallbackUsed: saliency.legacyFallbackUsed,
    threshold: selection.threshold,
    stats: {
      totalExperts: saliency.scoredExperts.length,
      prunedExperts: decisions.pruned.length,
      keptExperts: decisions.kept.length,
      blockedByLayerSafety: selection.blockedByLayerSafety
    },
    pruned: decisions.pruned,
    kept: decisions.kept
  };

  await observer.track('write_output', 'Wrote pruning plan to disk', async () =>
    components.writePlan(planPath, plan)
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
