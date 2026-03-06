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

  it('adds dataset-file and sample batching flags when set', () => {
    const args = __testOnly.buildMlxCollectArgs(
      {
        modelPath: '/tmp/model',
        outputDir: '/tmp/out',
        datasetFile: '/tmp/data.jsonl',
        datasetFormat: 'jsonl',
        datasetMessagesField: 'messages',
        maxSamples: 32,
        minSamples: 8,
        maxTokens: 512,
        sampleBatchSize: 4,
        packSamples: true,
        collectMode: 'reload_per_layer',
        lazyLoad: true
      },
      '/tmp/out/telemetry.json'
    );

    expect(args).toContain('--dataset-file');
    expect(args).toContain('/tmp/data.jsonl');
    expect(args).toContain('--dataset-format');
    expect(args).toContain('jsonl');
    expect(args).toContain('--dataset-messages-field');
    expect(args).toContain('messages');
    expect(args).toContain('--sample-batch-size');
    expect(args).toContain('4');
    expect(args).toContain('--pack-samples');
    expect(args).toContain('--lazy-load');
    expect(args).toContain('--collect-mode');
    expect(args).toContain('replay_per_layer');
  });

  it('throws when multiple input sources are set', () => {
    expect(() =>
      __testOnly.buildMlxCollectArgs(
        {
          ...baseConfig,
          datasetName: 'dummy/dataset'
        },
        '/tmp/out/telemetry.json'
      )
    ).toThrow(/exactly one/i);
  });

  it('throws when minSamples exceeds maxSamples', () => {
    expect(() =>
      __testOnly.buildMlxCollectArgs(
        {
          ...baseConfig,
          maxSamples: 4,
          minSamples: 5
        },
        '/tmp/out/telemetry.json'
      )
    ).toThrow(/minSamples/);
  });

  it('throws for invalid collect mode', () => {
    expect(() =>
      __testOnly.buildMlxCollectArgs(
        {
          ...baseConfig,
          collectMode: 'bogus' as never
        },
        '/tmp/out/telemetry.json'
      )
    ).toThrow(/collectMode/);
  });
});
