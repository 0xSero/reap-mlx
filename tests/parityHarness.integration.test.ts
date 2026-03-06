import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { runParityHarness } from '../src/core/index.js';

const createdDirs: string[] = [];

async function createTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'reap-mlx-parity-harness-'));
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

describe('parity harness', () => {
  it('reports exact parity when both sides use the same real telemetry fixture', async () => {
    const tempDir = await createTempDir();
    const root = path.resolve(__dirname, '..');
    const fixturePath = path.join(root, 'tests/fixtures/og-reap-real500.telemetry.json');
    const leftPath = path.join(tempDir, 'left.telemetry.json');
    const rightPath = path.join(tempDir, 'right.telemetry.json');

    await fs.copyFile(fixturePath, leftPath);
    await fs.copyFile(fixturePath, rightPath);

    const report = await runParityHarness({
      leftModelPath: leftPath,
      rightModelPath: rightPath,
      outputDir: path.join(tempDir, 'parity'),
      pruneMethod: 'reap',
      targetRatio: 0.5,
      minExpertsPerLayer: 1,
      nExpertsToPrunePerLayer: 2,
      allowLegacySaliency: false
    });

    expect(report.telemetry.normalizedHashEqual).toBe(true);
    expect(report.telemetry.differingExpertRows).toBe(0);
    expect(report.pruning.prunedExactMatch).toBe(true);
    expect(report.pruning.keptExactMatch).toBe(true);
    expect(report.pruning.thresholdEqual).toBe(true);
    expect(report.overallExactMatch).toBe(true);
    expect(report.pruning.onlyLeftPruned).toEqual([]);
    expect(report.pruning.onlyRightPruned).toEqual([]);
    await expect(fs.access(report.artifacts.reportJsonPath)).resolves.toBeUndefined();
    await expect(fs.access(report.artifacts.reportMarkdownPath)).resolves.toBeUndefined();
  });

  it('surfaces telemetry and prune drift when the right telemetry changes', async () => {
    const tempDir = await createTempDir();
    const root = path.resolve(__dirname, '..');
    const fixturePath = path.join(root, 'tests/fixtures/og-reap-real500.telemetry.json');
    const leftPath = path.join(tempDir, 'left.telemetry.json');
    const rightPath = path.join(tempDir, 'right.telemetry.json');

    await fs.copyFile(fixturePath, leftPath);
    await fs.copyFile(fixturePath, rightPath);

    const rightRaw = JSON.parse(await fs.readFile(rightPath, 'utf8')) as {
      experts: Array<{ layer: number; expert: number; reap: number }>;
    };
    const target = rightRaw.experts.find((entry) => entry.layer === 0 && entry.expert === 0);
    if (!target) {
      throw new Error('Failed to locate target expert for parity drift test');
    }
    target.reap = 99;
    await fs.writeFile(rightPath, `${JSON.stringify(rightRaw, null, 2)}\n`, 'utf8');

    const report = await runParityHarness({
      leftModelPath: leftPath,
      rightModelPath: rightPath,
      outputDir: path.join(tempDir, 'parity-drift'),
      pruneMethod: 'reap',
      targetRatio: 0.5,
      minExpertsPerLayer: 1,
      nExpertsToPrunePerLayer: 2,
      allowLegacySaliency: false
    });

    expect(report.telemetry.normalizedHashEqual).toBe(false);
    expect(report.telemetry.differingExpertRows).toBeGreaterThan(0);
    expect(report.telemetry.firstDifferingExpert).toMatchObject({
      layer: 0,
      expert: 0
    });
    expect(report.pruning.prunedExactMatch).toBe(false);
    expect(report.pruning.onlyLeftPruned.length + report.pruning.onlyRightPruned.length).toBeGreaterThan(0);
    expect(report.overallExactMatch).toBe(false);
  });
});
