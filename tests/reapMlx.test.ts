import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { runReapMlx, summarizeObservationLog } from '../src/core/index.js';
import { writeJsonAtomicSafe } from '../src/core/security.js';

const createdDirs: string[] = [];

async function createTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'reap-mlx-test-'));
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

describe('runReapMlx', () => {
  it('produces a pruning plan and observation log', async () => {
    const tempDir = await createTempDir();
    const modelPath = path.join(tempDir, 'telemetry.json');
    const outputDir = path.join(tempDir, 'out');

    await writeJsonAtomicSafe(modelPath, {
      modelName: 'unit-test-moe',
      experts: [
        { layer: 0, expert: 0, activationScore: 0.1, tokenCount: 500 },
        { layer: 0, expert: 1, activationScore: 1.1, tokenCount: 1200 },
        { layer: 1, expert: 0, activationScore: 0.2, tokenCount: 450 },
        { layer: 1, expert: 1, activationScore: 0.9, tokenCount: 1000 }
      ]
    });

    const plan = await runReapMlx({
      modelPath,
      outputDir,
      targetRatio: 0.5,
      calibrationRounds: 2,
      jobId: 'job-test'
    });

    expect(plan.jobId).toBe('job-test');
    expect(plan.stats.totalExperts).toBe(4);
    expect(plan.stats.prunedExperts).toBeGreaterThan(0);
    expect(plan.stats.keptExperts).toBeGreaterThan(0);

    const planFile = path.join(outputDir, 'pruning-plan.json');
    const logFile = path.join(outputDir, 'observation.log');

    const planRaw = await fs.readFile(planFile, 'utf8');
    const planJson = JSON.parse(planRaw) as { modelName: string };
    expect(planJson.modelName).toBe('unit-test-moe');

    const summary = await summarizeObservationLog(logFile);
    expect(summary.totalEvents).toBeGreaterThan(0);
    expect(summary.malformedLines).toBe(0);
    expect(summary.levels.error).toBe(0);
  });

  it('enforces layer safety by keeping at least one expert per layer', async () => {
    const tempDir = await createTempDir();
    const modelPath = path.join(tempDir, 'telemetry.json');
    const outputDir = path.join(tempDir, 'out');

    await writeJsonAtomicSafe(modelPath, {
      modelName: 'layer-safety',
      experts: [
        { layer: 0, expert: 0, activationScore: 0.1, tokenCount: 100 },
        { layer: 1, expert: 0, activationScore: 0.2, tokenCount: 100 },
        { layer: 2, expert: 0, activationScore: 0.3, tokenCount: 100 }
      ]
    });

    const plan = await runReapMlx({
      modelPath,
      outputDir,
      targetRatio: 0.9,
      calibrationRounds: 1
    });

    expect(plan.stats.prunedExperts).toBe(0);
    expect(plan.stats.keptExperts).toBe(3);
    expect(plan.achievedRatio).toBe(0);
  });
});
