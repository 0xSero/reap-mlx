import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { runReapMlx } from '../src/core/index.js';
import { writeJsonAtomicSafe } from '../src/core/security.js';
import type { ExpertSignal, PruneMethod } from '../src/core/types.js';

const createdDirs: string[] = [];

async function createTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'reap-mlx-parity-'));
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

interface TelemetryRow {
  layer: number;
  expert: number;
  frequency: number;
  eanSum: number;
  eanMean: number;
  weightedEanSum: number;
  reap: number;
  maxActivation: number;
}

const PRUNE_METHODS: PruneMethod[] = [
  'reap',
  'frequency',
  'ean_sum',
  'ean_mean',
  'weighted_ean_sum'
];

function toSet(rows: ReadonlyArray<{ layer: number; expert: number }>): Set<string> {
  return new Set(rows.map((row) => `${row.layer}:${row.expert}`));
}

function sortedKeyArray(values: Iterable<string>): string[] {
  return [...values].sort((left, right) => {
    const [leftLayer, leftExpert] = left.split(':').map((value) => Number(value));
    const [rightLayer, rightExpert] = right.split(':').map((value) => Number(value));

    if (leftLayer !== rightLayer) {
      return leftLayer - rightLayer;
    }

    return leftExpert - rightExpert;
  });
}

function percentile995(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const pos = ((sorted.length - 1) * 99.5) / 100;
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);

  const lowerValue = sorted[lower] ?? sorted[sorted.length - 1] ?? 0;
  const upperValue = sorted[upper] ?? sorted[sorted.length - 1] ?? 0;

  if (upper === lower) {
    return lowerValue;
  }

  return lowerValue + (upperValue - lowerValue) * (pos - lower);
}

function referencePreservedSet(
  telemetry: TelemetryRow[],
  preserveSuperExperts: boolean,
  preserveOutliers: boolean
): Set<string> {
  if (!preserveSuperExperts && !preserveOutliers) {
    return new Set<string>();
  }

  if (preserveSuperExperts && preserveOutliers) {
    throw new Error('Invalid parity test config: both preserve modes enabled');
  }

  const allLayers = [...new Set(telemetry.map((entry) => entry.layer))].sort(
    (left, right) => left - right
  );
  const layerCutoff = preserveOutliers
    ? allLayers.length
    : Math.floor(allLayers.length * 0.75);
  const eligibleLayers = new Set(allLayers.slice(0, layerCutoff));

  const maxValues = telemetry.map((entry) => entry.maxActivation);
  const threshold = Math.max(percentile995(maxValues), Math.max(...maxValues) / 10);

  const preserved = new Set<string>();

  for (const entry of telemetry) {
    if (!eligibleLayers.has(entry.layer)) {
      continue;
    }

    if (entry.maxActivation > threshold) {
      preserved.add(`${entry.layer}:${entry.expert}`);
    }
  }

  return preserved;
}

function metricValue(row: TelemetryRow, method: PruneMethod): number {
  if (method === 'frequency') {
    return row.frequency;
  }

  if (method === 'ean_sum') {
    return row.eanSum;
  }

  if (method === 'ean_mean') {
    return row.eanMean;
  }

  if (method === 'weighted_ean_sum') {
    return row.weightedEanSum;
  }

  return row.reap;
}

function referencePrunedSet(
  telemetry: TelemetryRow[],
  method: PruneMethod,
  options: {
    targetRatio: number;
    minExpertsPerLayer: number;
    nExpertsToPrunePerLayer?: number;
    preserveSuperExperts: boolean;
    preserveOutliers: boolean;
  }
): Set<string> {
  const preserved = referencePreservedSet(
    telemetry,
    options.preserveSuperExperts,
    options.preserveOutliers
  );

  const grouped = new Map<number, TelemetryRow[]>();
  for (const entry of telemetry) {
    const layerRows = grouped.get(entry.layer);
    if (layerRows) {
      layerRows.push(entry);
    } else {
      grouped.set(entry.layer, [entry]);
    }
  }

  const pruned = new Set<string>();

  for (const layerRows of grouped.values()) {
    const sorted = [...layerRows].sort((left, right) => {
      const leftSignal = preserved.has(`${left.layer}:${left.expert}`)
        ? Number.POSITIVE_INFINITY
        : metricValue(left, method);
      const rightSignal = preserved.has(`${right.layer}:${right.expert}`)
        ? Number.POSITIVE_INFINITY
        : metricValue(right, method);

      if (leftSignal !== rightSignal) {
        return leftSignal - rightSignal;
      }

      return left.expert - right.expert;
    });

    const layerRequested =
      typeof options.nExpertsToPrunePerLayer === 'number'
        ? Math.min(sorted.length, options.nExpertsToPrunePerLayer)
        : Math.floor(sorted.length * options.targetRatio);

    const layerMaxPrunableBySafety = Math.max(0, sorted.length - options.minExpertsPerLayer);
    const layerMaxPrunableByPreserve = sorted.filter(
      (entry) => !preserved.has(`${entry.layer}:${entry.expert}`)
    ).length;

    const layerTarget =
      typeof options.nExpertsToPrunePerLayer === 'number'
        ? Math.min(layerRequested, layerMaxPrunableByPreserve)
        : Math.min(layerRequested, layerMaxPrunableBySafety, layerMaxPrunableByPreserve);

    for (const row of sorted.slice(0, layerTarget)) {
      pruned.add(`${row.layer}:${row.expert}`);
    }
  }

  return pruned;
}

function telemetryToExpertSignals(rows: TelemetryRow[]): ExpertSignal[] {
  return rows.map((row) => ({
    layer: row.layer,
    expert: row.expert,
    frequency: row.frequency,
    eanSum: row.eanSum,
    eanMean: row.eanMean,
    weightedEanSum: row.weightedEanSum,
    reap: row.reap,
    maxActivation: row.maxActivation,
    tokenCount: 100,
    activeTokenCount: 100
  }));
}

async function actualPrunedSet(
  telemetry: TelemetryRow[],
  method: PruneMethod,
  options: {
    targetRatio: number;
    minExpertsPerLayer: number;
    nExpertsToPrunePerLayer?: number;
    preserveSuperExperts: boolean;
    preserveOutliers: boolean;
  }
): Promise<Set<string>> {
  const tempDir = await createTempDir();
  const modelPath = path.join(tempDir, 'telemetry.json');
  const outputDir = path.join(tempDir, 'out');

  await writeJsonAtomicSafe(modelPath, {
    modelName: 'parity-test-moe',
    experts: telemetryToExpertSignals(telemetry)
  });

  const plan = await runReapMlx({
    modelPath,
    outputDir,
    targetRatio: options.targetRatio,
    minExpertsPerLayer: options.minExpertsPerLayer,
    pruneMethod: method,
    ...(typeof options.nExpertsToPrunePerLayer === 'number'
      ? { nExpertsToPrunePerLayer: options.nExpertsToPrunePerLayer }
      : {}),
    preserveSuperExperts: options.preserveSuperExperts,
    preserveOutliers: options.preserveOutliers,
    allowLegacySaliency: false
  });

  return toSet(plan.pruned);
}

function syntheticTelemetry(): TelemetryRow[] {
  return [
    { layer: 0, expert: 0, frequency: 5, eanSum: 30, eanMean: 3, weightedEanSum: 7, reap: 0.40, maxActivation: 4 },
    { layer: 0, expert: 1, frequency: 2, eanSum: 10, eanMean: 1, weightedEanSum: 2, reap: 0.10, maxActivation: 60 },
    { layer: 0, expert: 2, frequency: 8, eanSum: 40, eanMean: 4, weightedEanSum: 11, reap: 0.80, maxActivation: 6 },
    { layer: 0, expert: 3, frequency: 1, eanSum: 8, eanMean: 0.8, weightedEanSum: 1, reap: 0.05, maxActivation: 5 },

    { layer: 1, expert: 0, frequency: 3, eanSum: 22, eanMean: 2.2, weightedEanSum: 5, reap: 0.21, maxActivation: 7 },
    { layer: 1, expert: 1, frequency: 9, eanSum: 45, eanMean: 4.5, weightedEanSum: 15, reap: 0.95, maxActivation: 8 },
    { layer: 1, expert: 2, frequency: 2, eanSum: 15, eanMean: 1.5, weightedEanSum: 3, reap: 0.15, maxActivation: 55 },
    { layer: 1, expert: 3, frequency: 6, eanSum: 28, eanMean: 2.8, weightedEanSum: 8, reap: 0.33, maxActivation: 9 },

    { layer: 2, expert: 0, frequency: 4, eanSum: 20, eanMean: 2, weightedEanSum: 5, reap: 0.25, maxActivation: 10 },
    { layer: 2, expert: 1, frequency: 7, eanSum: 32, eanMean: 3.2, weightedEanSum: 9, reap: 0.52, maxActivation: 12 },
    { layer: 2, expert: 2, frequency: 2, eanSum: 11, eanMean: 1.1, weightedEanSum: 2, reap: 0.11, maxActivation: 11 },
    { layer: 2, expert: 3, frequency: 6, eanSum: 26, eanMean: 2.6, weightedEanSum: 7, reap: 0.31, maxActivation: 13 },

    { layer: 3, expert: 0, frequency: 1, eanSum: 9, eanMean: 0.9, weightedEanSum: 2, reap: 0.08, maxActivation: 80 },
    { layer: 3, expert: 1, frequency: 8, eanSum: 35, eanMean: 3.5, weightedEanSum: 10, reap: 0.60, maxActivation: 15 },
    { layer: 3, expert: 2, frequency: 3, eanSum: 14, eanMean: 1.4, weightedEanSum: 4, reap: 0.19, maxActivation: 16 },
    { layer: 3, expert: 3, frequency: 5, eanSum: 24, eanMean: 2.4, weightedEanSum: 6, reap: 0.29, maxActivation: 14 }
  ];
}

describe('Cerebras parity pruning selection', () => {
  it('matches reference implementation for ratio mode across methods and preserve settings', async () => {
    const telemetry = syntheticTelemetry();

    const preserveModes = [
      { preserveSuperExperts: false, preserveOutliers: false },
      { preserveSuperExperts: true, preserveOutliers: false },
      { preserveSuperExperts: false, preserveOutliers: true }
    ];

    for (const method of PRUNE_METHODS) {
      for (const preserveMode of preserveModes) {
        const options = {
          targetRatio: 0.5,
          minExpertsPerLayer: 1,
          ...preserveMode
        };

        const expected = referencePrunedSet(telemetry, method, options);
        const actual = await actualPrunedSet(telemetry, method, options);

        expect(sortedKeyArray(actual)).toEqual(sortedKeyArray(expected));
      }
    }
  });

  it('matches reference implementation for fixed prune count mode across methods and preserve settings', async () => {
    const telemetry = syntheticTelemetry();

    const preserveModes = [
      { preserveSuperExperts: false, preserveOutliers: false },
      { preserveSuperExperts: true, preserveOutliers: false },
      { preserveSuperExperts: false, preserveOutliers: true }
    ];

    for (const method of PRUNE_METHODS) {
      for (const preserveMode of preserveModes) {
        const options = {
          targetRatio: 0.1,
          minExpertsPerLayer: 1,
          nExpertsToPrunePerLayer: 2,
          ...preserveMode
        };

        const expected = referencePrunedSet(telemetry, method, options);
        const actual = await actualPrunedSet(telemetry, method, options);

        expect(sortedKeyArray(actual)).toEqual(sortedKeyArray(expected));
      }
    }
  });
});
