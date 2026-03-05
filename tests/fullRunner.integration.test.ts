import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { afterEach, describe, expect, it } from 'vitest';

const execFileAsync = promisify(execFile);
const createdDirs: string[] = [];

async function createTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'reap-mlx-full-test-'));
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

function toLayerMap(rows: Array<{ layer: number; expert: number }>): Map<number, number[]> {
  const grouped = new Map<number, number[]>();

  for (const row of rows) {
    const list = grouped.get(row.layer);
    if (list) {
      list.push(row.expert);
    } else {
      grouped.set(row.layer, [row.expert]);
    }
  }

  for (const [layer, values] of grouped.entries()) {
    grouped.set(layer, [...values].sort((left, right) => left - right));
  }

  return grouped;
}

describe('full runner harness', () => {
  it('reproduces OG REAP fixture prune set deterministically', { timeout: 30000 }, async () => {
    const tempDir = await createTempDir();
    const root = path.resolve(__dirname, '..');
    const telemetrySource = path.join(root, 'tests/fixtures/og-reap-real500.telemetry.json');

    const telemetryPath = path.join(tempDir, 'telemetry.json');
    await fs.copyFile(telemetrySource, telemetryPath);

    const runArgs = [
      path.join(root, 'dist/cli/index.js'),
      'run',
      '--model',
      telemetryPath,
      '--ratio',
      '0.5',
      '--min-experts',
      '1',
      '--prune-method',
      'reap',
      '--n-experts-to-prune-per-layer',
      '2',
      '--no-legacy',
      '--json'
    ];

    const run1 = await execFileAsync('node', [...runArgs, '--output', path.join(tempDir, 'out-1')]);
    const run2 = await execFileAsync('node', [...runArgs, '--output', path.join(tempDir, 'out-2')]);

    const plan1 = JSON.parse(run1.stdout) as {
      threshold: number;
      pruned: Array<{ layer: number; expert: number }>;
    };
    const plan2 = JSON.parse(run2.stdout) as {
      threshold: number;
      pruned: Array<{ layer: number; expert: number }>;
    };

    const map1 = toLayerMap(plan1.pruned);
    const map2 = toLayerMap(plan2.pruned);

    expect(map1.get(0)).toEqual([0, 3]);
    expect(map1.get(1)).toEqual([1, 2]);

    expect(map2.get(0)).toEqual([0, 3]);
    expect(map2.get(1)).toEqual([1, 2]);

    expect(map2.get(0)).toEqual(map1.get(0));
    expect(map2.get(1)).toEqual(map1.get(1));
    expect(plan2.threshold).toBeCloseTo(plan1.threshold, 12);
  });
});
