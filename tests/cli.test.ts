import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { afterEach, describe, expect, it } from 'vitest';

const execFileAsync = promisify(execFile);
const createdDirs: string[] = [];

async function createTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'reap-mlx-cli-test-'));
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

describe('cli', () => {
  it('documents layer-wise and batch-size flags in help output', async () => {
    const root = path.resolve(__dirname, '..');

    const helpResult = await execFileAsync('node', [path.join(root, 'dist/cli/index.js'), 'help']);

    expect(helpResult.stdout).toContain('--layer-wise');
    expect(helpResult.stdout).toContain('--batch-size <1..8192>');
  });

  it('initializes telemetry then runs and observes it', async () => {
    const tempDir = await createTempDir();
    const telemetryPath = path.join(tempDir, 'telemetry.json');
    const outputDir = path.join(tempDir, 'out');
    const root = path.resolve(__dirname, '..');

    await execFileAsync('node', [
      path.join(root, 'dist/cli/index.js'),
      'init',
      '--output',
      telemetryPath,
      '--layers',
      '4',
      '--experts',
      '6',
      '--seed',
      '123'
    ]);

    const runResult = await execFileAsync('node', [
      path.join(root, 'dist/cli/index.js'),
      'run',
      '--model',
      telemetryPath,
      '--output',
      outputDir,
      '--ratio',
      '0.4'
    ]);

    expect(runResult.stdout).toContain('pruned experts:');

    const observeResult = await execFileAsync('node', [
      path.join(root, 'dist/cli/index.js'),
      'observe',
      '--file',
      path.join(outputDir, 'observation.log'),
      '--json'
    ]);

    const parsed = JSON.parse(observeResult.stdout) as { totalEvents: number };
    expect(parsed.totalEvents).toBeGreaterThan(0);
  });

  it('supports run with fixed prune count and no ratio', async () => {
    const tempDir = await createTempDir();
    const telemetryPath = path.join(tempDir, 'telemetry.json');
    const outputDir = path.join(tempDir, 'out-fixed');
    const root = path.resolve(__dirname, '..');

    await execFileAsync('node', [
      path.join(root, 'dist/cli/index.js'),
      'init',
      '--output',
      telemetryPath,
      '--layers',
      '2',
      '--experts',
      '4',
      '--seed',
      '456'
    ]);

    const runResult = await execFileAsync('node', [
      path.join(root, 'dist/cli/index.js'),
      'run',
      '--model',
      telemetryPath,
      '--output',
      outputDir,
      '--n-experts-to-prune-per-layer',
      '1'
    ]);

    expect(runResult.stdout).toContain('pruned experts:');
  });

  it('validates collect --batch-size flag', async () => {
    const tempDir = await createTempDir();
    const outputDir = path.join(tempDir, 'collect-out');
    const root = path.resolve(__dirname, '..');

    await expect(
      execFileAsync('node', [
        path.join(root, 'dist/cli/index.js'),
        'collect',
        '--model',
        '/tmp/model-not-used',
        '--output',
        outputDir,
        '--prompt',
        'hello',
        '--batch-size',
        '0'
      ])
    ).rejects.toMatchObject({
      stderr: expect.stringContaining('batch-size')
    });
  });

  it('validates full --batch-size flag', async () => {
    const tempDir = await createTempDir();
    const outputDir = path.join(tempDir, 'full-out');
    const root = path.resolve(__dirname, '..');

    await expect(
      execFileAsync('node', [
        path.join(root, 'dist/cli/index.js'),
        'full',
        '--model',
        '/tmp/model-not-used',
        '--output',
        outputDir,
        '--dataset',
        'dummy/dataset',
        '--batch-size',
        '0'
      ])
    ).rejects.toMatchObject({
      stderr: expect.stringContaining('batch-size')
    });
  });
});
