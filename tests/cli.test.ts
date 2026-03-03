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
});
