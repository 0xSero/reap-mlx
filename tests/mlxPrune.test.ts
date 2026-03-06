import { existsSync, promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { applyPruningPlanToMlxModel } from '../src/core/index.js';
import { writeJsonAtomicSafe } from '../src/core/security.js';

const createdDirs: string[] = [];
const DRY_RUN_MODEL_PATH =
  process.env.REAP_MLX_TEST_MODEL_PATH ??
  '/Users/sero/projects/quantforge/models/qwen1.5-moe-a2.7b-chat-4bit';
const HAS_DRY_RUN_MODEL = existsSync(DRY_RUN_MODEL_PATH);

async function createTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'reap-mlx-apply-test-'));
  createdDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(
    createdDirs.splice(0).map(async (dir) => {
      await fs.rm(dir, { recursive: true, force: true });
    })
  );
});

describe('applyPruningPlanToMlxModel', () => {
  it.skipIf(!HAS_DRY_RUN_MODEL)(
    'supports dry-run validation path',
    async () => {
      const tempDir = await createTempDir();
      const planPath = path.join(tempDir, 'plan.json');

      await writeJsonAtomicSafe(planPath, {
        version: '1.0',
        createdAt: new Date().toISOString(),
        jobId: 'job-dryrun',
        modelName: 'dummy-model',
        targetRatio: 0.5,
        achievedRatio: 0.5,
        calibrationRounds: 1,
        saliencyMethod: 'reap',
        minExpertsPerLayer: 1,
        legacyFallbackUsed: false,
        threshold: 0.1,
        stats: {
          totalExperts: 4,
          prunedExperts: 2,
          keptExperts: 2,
          blockedByLayerSafety: 0
        },
        pruned: [
          {
            layer: 0,
            expert: 0,
            signal: 0.01,
            rank: 1,
            reason: 'low_signal_pruned',
            saliencySource: 'weighted_activation_sum',
            activeTokenCount: 10
          },
          {
            layer: 1,
            expert: 0,
            signal: 0.02,
            rank: 2,
            reason: 'low_signal_pruned',
            saliencySource: 'weighted_activation_sum',
            activeTokenCount: 12
          }
        ],
        kept: [
          {
            layer: 0,
            expert: 1,
            signal: 0.9,
            rank: 3,
            reason: 'high_signal_retained',
            saliencySource: 'weighted_activation_sum',
            activeTokenCount: 9
          },
          {
            layer: 1,
            expert: 1,
            signal: 1.0,
            rank: 4,
            reason: 'high_signal_retained',
            saliencySource: 'weighted_activation_sum',
            activeTokenCount: 11
          }
        ]
      });

      const result = await applyPruningPlanToMlxModel({
        modelPath: DRY_RUN_MODEL_PATH,
        planPath,
        outputDir: path.join(tempDir, 'out'),
        dryRun: true
      });

      expect(result.layersPatched).toBeGreaterThan(0);
      expect(result.expertsBefore).toBeGreaterThanOrEqual(result.expertsAfter);
      expect(result.pruningPlanJobId).toBe('job-dryrun');
    },
    60000
  );
});
