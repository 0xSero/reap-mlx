import { describe, expect, it } from 'vitest';
import { __testOnly } from '../src/core/mlxCollector.js';

const baseConfig = {
  modelPath: '/tmp/model',
  outputDir: '/tmp/out',
  prompt: 'hello',
  maxTokens: 256
};

describe('mlxCollector args', () => {
  it('adds --layer-wise when enabled', () => {
    const args = __testOnly.buildMlxCollectArgs(
      {
        ...baseConfig,
        layerWise: true
      },
      '/tmp/out/telemetry.json'
    );

    expect(args).toContain('--layer-wise');
  });

  it('adds --batch-size when set', () => {
    const args = __testOnly.buildMlxCollectArgs(
      {
        ...baseConfig,
        batchSize: 64
      },
      '/tmp/out/telemetry.json'
    );

    const index = args.indexOf('--batch-size');
    expect(index).toBeGreaterThan(-1);
    expect(args[index + 1]).toBe('64');
  });

  it('throws for invalid batch size', () => {
    expect(() =>
      __testOnly.buildMlxCollectArgs(
        {
          ...baseConfig,
          batchSize: 0
        },
        '/tmp/out/telemetry.json'
      )
    ).toThrow(/batchSize/);
  });
});
