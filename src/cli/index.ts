#!/usr/bin/env node

import { randomUUID } from 'node:crypto';
import { constants, promises as fs } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  applyPruningPlanToMlxModel,
  collectTelemetryWithMlx,
  probeMlxModelCoherence,
  runParityHarness,
  runReapMlx,
  summarizeObservationLog
} from '../core/index.js';
import type { CollectMode, DatasetFormat, MlxCollectConfig } from '../core/mlxCollector.js';
import {
  assertFiniteNumber,
  assertInteger,
  ensureSecureDir,
  writeJsonAtomicSafe
} from '../core/security.js';
import type {
  ModelTelemetry,
  ObservationSummary,
  PruneMethod,
  PruningPlan
} from '../core/types.js';

type CliOptions = Map<string, string | boolean>;

interface ParsedCli {
  command: string;
  options: CliOptions;
  positionals: string[];
}

interface CollectCliConfig extends MlxCollectConfig {
  outputDir: string;
  modelPath: string;
}

function printUsage(): void {
  process.stdout.write(`reap-mlx - secure REAP pruning planner for MLX telemetry

Usage:
  reap-mlx <command> [options]

Commands:
  run       Build pruning plan from telemetry JSON
  parity    Compare two telemetry files under one exact pruning config
  collect   Collect REAP telemetry from MLX model (prompt or dataset)
  full      Run full pipeline: collect -> run -> apply
  apply     Apply pruning plan to an MLX checkpoint
  probe     Prompt an MLX model and print the completion
  observe   Summarize observation log
  init      Generate synthetic telemetry JSON
  help      Show this help
  version   Print version

run options:
  --model <file>                    Telemetry JSON input file
  --output <dir>                    Output directory for plan + logs
  --ratio <0..0.95>                 Target prune ratio
  --calibration <1..25>             Calibration rounds (default: 2)
  --min-experts <1..128>            Minimum experts preserved per layer (default: 1)
  --prune-method <name>             Pruning metric: reap|reap_l2|frequency|weighted_frequency_sum|ean_sum|ean_mean|ean_ca|weighted_ean_sum|weighted_ean_sum_l2|max_activations
  --n-experts-to-prune-per-layer <n> Prune exactly n experts per layer (bounded by layer size)
  --preserve-super-experts          Preserve super experts from pruning mask
  --preserve-outliers               Preserve outlier experts (include all layers)
  --no-legacy                       Require REAP saliency fields (disable activationScore fallback)
  --job-id <id>                     Optional job id
  --observation <name>              Observation filename (default: observation.log)
  --json                            Output full result as JSON

parity options:
  --left <file>                     Left telemetry JSON input file
  --right <file>                    Right telemetry JSON input file
  --output <dir>                    Output directory for plans + parity report
  --ratio <0..0.95>                 Target prune ratio
  --calibration <1..25>             Calibration rounds (default: 2)
  --min-experts <1..128>            Minimum experts preserved per layer (default: 1)
  --prune-method <name>             Pruning metric: reap|reap_l2|frequency|weighted_frequency_sum|ean_sum|ean_mean|ean_ca|weighted_ean_sum|weighted_ean_sum_l2|max_activations
  --n-experts-to-prune-per-layer <n> Prune exactly n experts per layer (bounded by layer size)
  --preserve-super-experts          Preserve super experts from pruning mask
  --preserve-outliers               Preserve outlier experts (include all layers)
  --no-legacy                       Require REAP saliency fields (disable activationScore fallback)
  --require-identical-telemetry     Exit non-zero unless normalized telemetry hashes are identical
  --json                            Output full parity report as JSON

collect options:
  --model <dir>            Local MLX model directory
  --output <dir>           Output directory for telemetry JSON
  --prompt <text>          Single calibration text
  --dataset <name>         HuggingFace dataset name (full-runner mode)
  --dataset-file <path>    Local calibration dataset file
  --dataset-format <fmt>   auto|json|jsonl|csv|parquet|text (dataset-file only)
  --dataset-split <name>   Dataset split (default: train)
  --dataset-text-field <f> Field path for text (default: auto common fields)
  --dataset-messages-field <f> Field path for chat messages array
  --max-samples <n>        Max dataset samples to aggregate (default: 100)
  --min-samples <n>        Require at least n usable samples (default: 1)
  --max-tokens <1..16384>  Per-sample token cap (default: 256)
  --sample-batch-size <1..1024> Batch multiple samples/conversations together
  --pack-samples           Pack multiple independent samples into max-tokens windows
  --layers <spec>          Optional layer filter (e.g. 0-3,8,10)
  --renorm-topk            Renormalize top-k gate weights to sum to 1
  --layer-wise             Enable layer-wise collection mode
  --collect-mode <name>    single_pass|replay_per_layer|reload_per_layer
  --batch-size <1..8192>   Token chunk size for collection batching
  --lazy-load              Ask MLX to lazily materialize weights during load
  --python <bin>           Python binary (default: python3)

full options:
  --model <dir>            Local MLX model directory
  --output <dir>           Pipeline output directory
  --prompt <text>          Single calibration text
  --dataset <name>         HuggingFace dataset name
  --dataset-file <path>    Local calibration dataset file
  --dataset-format <fmt>   auto|json|jsonl|csv|parquet|text (dataset-file only)
  --dataset-split <name>   Dataset split (default: train)
  --dataset-text-field <f> Field path for text (default: auto common fields)
  --dataset-messages-field <f> Field path for chat messages array
  --max-samples <n>        Max dataset samples to aggregate (default: 100)
  --min-samples <n>        Require at least n usable samples (default: 1)
  --max-tokens <1..16384>  Per-sample token cap (default: 256)
  --sample-batch-size <1..1024> Batch multiple samples/conversations together
  --pack-samples           Pack multiple independent samples into max-tokens windows
  --layers <spec>          Optional layer filter (e.g. 0-3,8,10)
  --renorm-topk            Renormalize top-k gate weights to sum to 1
  --layer-wise             Enable layer-wise collection mode
  --collect-mode <name>    single_pass|replay_per_layer|reload_per_layer
  --batch-size <1..8192>   Token chunk size for collection batching
  --lazy-load              Ask MLX to lazily materialize weights during load
  --ratio <0..0.95>        Target prune ratio (default: 0.5)
  --min-experts <1..128>   Minimum experts preserved per layer (default: 1)
  --prune-method <name>    Pruning metric: reap|reap_l2|frequency|weighted_frequency_sum|ean_sum|ean_mean|ean_ca|weighted_ean_sum|weighted_ean_sum_l2|max_activations
  --n-experts-to-prune-per-layer <n> Prune exactly n experts per layer
  --preserve-super-experts Preserve super experts from pruning mask
  --preserve-outliers      Preserve outlier experts (include all layers)
  --no-legacy              Require REAP saliency fields
  --python <bin>           Python binary (default: python3)
  --dry-run                Validate apply step without writing pruned model
  --json                   Output pipeline result as JSON

apply options:
  --model <dir>            Source MLX model directory
  --plan <file>            Pruning plan JSON from reap-mlx run
  --output <dir>           Output pruned model directory
  --python <bin>           Python binary (default: python3)
  --dry-run                Validate and simulate patch without writing model

probe options:
  --model <dir>            MLX model directory to prompt
  --prompt <text>          Prompt text for a quick coherence check
  --max-tokens <1..4096>   Generation cap (default: 80)
  --temperature <0..5>     Sampling temperature (default: 0.2)
  --python <bin>           Python binary (default: python3)

observe options:
  --file <path>            Observation log file
  --json                   Output summary as JSON

init options:
  --output <file>          Output telemetry JSON file
  --model-name <name>      Model name (default: synthetic-moe)
  --layers <1..512>        Layer count (default: 8)
  --experts <2..512>       Experts per layer (default: 8)
  --seed <int>             Seed (default: random)
`);
}

function parseArgs(argv: readonly string[]): ParsedCli {
  const args = argv.slice(2);
  const commandToken = args.at(0);
  const command =
    !commandToken || commandToken.startsWith('-') ? 'help' : commandToken.toLowerCase();

  const options: CliOptions = new Map();
  const positionals: string[] = [];

  for (let index = 1; index < args.length; index += 1) {
    const token = args.at(index);

    if (!token) {
      continue;
    }

    if (token === '--') {
      const tail = args.slice(index + 1);
      positionals.push(...tail);
      break;
    }

    if (!token.startsWith('--')) {
      positionals.push(token);
      continue;
    }

    const equalIndex = token.indexOf('=');
    if (equalIndex >= 0) {
      const key = token.slice(2, equalIndex);
      const value = token.slice(equalIndex + 1);
      options.set(key, value);
      continue;
    }

    const key = token.slice(2);
    const next = args.at(index + 1);

    if (!next || next.startsWith('--')) {
      options.set(key, true);
      continue;
    }

    options.set(key, next);
    index += 1;
  }

  return {
    command,
    options,
    positionals
  };
}

function optionString(options: CliOptions, key: string): string | undefined {
  const value = options.get(key);

  if (typeof value === 'string') {
    const normalized = value.trim();
    return normalized.length > 0 ? normalized : undefined;
  }

  return undefined;
}

function requiredOption(options: CliOptions, key: string): string {
  const value = optionString(options, key);
  if (!value) {
    throw new Error(`Missing required option --${key}`);
  }
  return value;
}

function optionBoolean(options: CliOptions, key: string): boolean {
  return options.get(key) === true;
}

function parsePruneMethod(value: string | undefined): PruneMethod | undefined {
  if (!value) {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  const allowed = new Set<PruneMethod>([
    'reap',
    'reap_l2',
    'frequency',
    'weighted_frequency_sum',
    'ean_sum',
    'ean_mean',
    'ean_ca',
    'weighted_ean_sum',
    'weighted_ean_sum_l2',
    'max_activations'
  ]);

  if (!allowed.has(normalized as PruneMethod)) {
    throw new Error(
      `Invalid --prune-method value: ${value}. Expected one of reap, reap_l2, frequency, weighted_frequency_sum, ean_sum, ean_mean, ean_ca, weighted_ean_sum, weighted_ean_sum_l2, max_activations`
    );
  }

  return normalized as PruneMethod;
}

function parseDatasetFormat(value: string | undefined): DatasetFormat | undefined {
  if (!value) {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  const allowed = new Set<DatasetFormat>(['auto', 'json', 'jsonl', 'csv', 'parquet', 'text']);

  if (!allowed.has(normalized as DatasetFormat)) {
    throw new Error(
      `Invalid --dataset-format value: ${value}. Expected one of auto, json, jsonl, csv, parquet, text`
    );
  }

  return normalized as DatasetFormat;
}

function parseCollectMode(value: string | undefined): CollectMode | undefined {
  if (!value) {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  const allowed = new Set<CollectMode>([
    'single_pass',
    'replay_per_layer',
    'reload_per_layer'
  ]);

  if (!allowed.has(normalized as CollectMode)) {
    throw new Error(
      `Invalid --collect-mode value: ${value}. Expected one of single_pass, replay_per_layer, reload_per_layer`
    );
  }

  return normalized as CollectMode;
}

function buildCollectConfigFromOptions(options: CliOptions, base: {
  modelPath: string;
  outputDir: string;
}): CollectCliConfig {
  const prompt = optionString(options, 'prompt');
  const datasetName = optionString(options, 'dataset');
  const datasetFile = optionString(options, 'dataset-file');
  const inputSources = [prompt, datasetName, datasetFile].filter(
    (value): value is string => typeof value === 'string' && value.length > 0
  );

  if (inputSources.length !== 1) {
    throw new Error('Use exactly one of --prompt, --dataset, or --dataset-file');
  }

  const datasetFormat = parseDatasetFormat(optionString(options, 'dataset-format'));
  if (datasetFormat && !datasetFile) {
    throw new Error('--dataset-format requires --dataset-file');
  }

  const datasetSplit = optionString(options, 'dataset-split') ?? 'train';
  const datasetTextField = optionString(options, 'dataset-text-field');
  const datasetMessagesField = optionString(options, 'dataset-messages-field');
  const maxSamples = assertInteger(
    optionString(options, 'max-samples') ?? 100,
    'max-samples',
    1,
    100000
  );
  const minSamples = assertInteger(
    optionString(options, 'min-samples') ?? 1,
    'min-samples',
    1,
    100000
  );
  if (minSamples > maxSamples) {
    throw new Error('min-samples cannot exceed max-samples');
  }

  const maxTokens = assertInteger(
    optionString(options, 'max-tokens') ?? 256,
    'max-tokens',
    1,
    16384
  );
  const sampleBatchSize = assertInteger(
    optionString(options, 'sample-batch-size') ?? 1,
    'sample-batch-size',
    1,
    1024
  );
  const includeLayers = optionString(options, 'layers');
  const renormTopK = optionBoolean(options, 'renorm-topk');
  const layerWise = optionBoolean(options, 'layer-wise');
  const collectMode = parseCollectMode(optionString(options, 'collect-mode'));
  if (layerWise && collectMode === 'single_pass') {
    throw new Error('--layer-wise conflicts with --collect-mode single_pass');
  }

  const batchSizeOption = optionString(options, 'batch-size');
  const batchSize =
    typeof batchSizeOption === 'string'
      ? assertInteger(batchSizeOption, 'batch-size', 1, 8192)
      : undefined;
  const packSamples = optionBoolean(options, 'pack-samples');
  const lazyLoad = optionBoolean(options, 'lazy-load');
  const pythonBin = optionString(options, 'python');

  return {
    modelPath: base.modelPath,
    outputDir: base.outputDir,
    ...(prompt ? { prompt } : {}),
    ...(datasetName ? { datasetName } : {}),
    ...(datasetFile ? { datasetFile } : {}),
    ...(datasetFormat ? { datasetFormat } : {}),
    ...(datasetSplit ? { datasetSplit } : {}),
    ...(datasetTextField ? { datasetTextField } : {}),
    ...(datasetMessagesField ? { datasetMessagesField } : {}),
    maxSamples,
    minSamples,
    maxTokens,
    sampleBatchSize,
    ...(includeLayers ? { includeLayers } : {}),
    ...(renormTopK ? { renormTopK: true } : {}),
    ...(layerWise ? { layerWise: true } : {}),
    ...(collectMode ? { collectMode } : {}),
    ...(typeof batchSize === 'number' ? { batchSize } : {}),
    ...(packSamples ? { packSamples: true } : {}),
    ...(lazyLoad ? { lazyLoad: true } : {}),
    ...(pythonBin ? { pythonBin } : {})
  };
}

function printTelemetryCollectionSummary(metadata: ModelTelemetry['metadata'] | undefined): void {
  if (!metadata) {
    return;
  }

  const stringField = (key: string, label: string) => {
    const value = metadata[key];
    if (typeof value === 'string' && value.length > 0) {
      process.stdout.write(`${label}: ${value}\n`);
    }
  };
  const numberField = (key: string, label: string) => {
    const value = metadata[key];
    if (typeof value === 'number') {
      process.stdout.write(`${label}: ${value}\n`);
    }
  };
  const booleanField = (key: string, label: string) => {
    const value = metadata[key];
    if (typeof value === 'boolean') {
      process.stdout.write(`${label}: ${value ? 'enabled' : 'disabled'}\n`);
    }
  };

  stringField('inputMode', 'input mode');
  stringField('dataset', 'dataset');
  stringField('datasetFile', 'dataset file');
  stringField('collectMode', 'collect mode');
  numberField('scannedSamples', 'scanned samples');
  numberField('processedSamples', 'processed samples');
  numberField('skippedSamples', 'skipped samples');
  numberField('modelTokenCount', 'model tokens');
  numberField('packedSequences', 'packed sequences');
  numberField('sampleBatches', 'sample batches');
  booleanField('layerWise', 'layer-wise');
  booleanField('packSamples', 'pack samples');
  booleanField('lazyLoad', 'lazy load');

  const batchSize = metadata.batchSize;
  if (typeof batchSize === 'number' && batchSize > 0) {
    process.stdout.write(`token batch size: ${batchSize}\n`);
  }

  const sampleBatchSize = metadata.sampleBatchSize;
  if (typeof sampleBatchSize === 'number' && sampleBatchSize > 0) {
    process.stdout.write(`sample batch size: ${sampleBatchSize}\n`);
  }
}

function createRng(seedInput: number): () => number {
  let state = seedInput >>> 0;

  if (state === 0) {
    state = 0x9e3779b9;
  }

  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

function buildSyntheticTelemetry(options: {
  modelName: string;
  layers: number;
  experts: number;
  seed: number;
}): ModelTelemetry {
  const random = createRng(options.seed);
  const experts = [] as ModelTelemetry['experts'];

  for (let layer = 0; layer < options.layers; layer += 1) {
    for (let expert = 0; expert < options.experts; expert += 1) {
      const activationScore = Number((random() * 1.4 + 0.05).toFixed(6));
      const tokenCount = Math.floor(random() * 3000) + 200;
      experts.push({
        layer,
        expert,
        activationScore,
        tokenCount
      });
    }
  }

  return {
    modelName: options.modelName,
    experts,
    metadata: {
      source: 'reap-mlx synthetic telemetry',
      seed: options.seed
    }
  };
}

function printRunSummary(plan: PruningPlan): void {
  process.stdout.write(
    [
      `jobId: ${plan.jobId}`,
      `model: ${plan.modelName}`,
      `target ratio: ${plan.targetRatio}`,
      `method: ${plan.saliencyMethod}`,
      `achieved ratio: ${plan.achievedRatio}`,
      `pruned experts: ${plan.stats.prunedExperts}/${plan.stats.totalExperts}`,
      `threshold: ${plan.threshold}`,
      `blocked by layer safety: ${plan.stats.blockedByLayerSafety}`
    ].join('\n') + '\n'
  );
}

function printObservationSummary(summary: ObservationSummary): void {
  process.stdout.write(
    [
      `jobId: ${summary.jobId ?? 'unknown'}`,
      `events: ${summary.totalEvents}`,
      `malformed lines: ${summary.malformedLines}`,
      `duration ms: ${summary.totalDurationMs}`,
      `levels: info=${summary.levels.info}, warn=${summary.levels.warn}, error=${summary.levels.error}`
    ].join('\n') + '\n'
  );
}

async function ensureFileReadable(filePath: string): Promise<void> {
  const absolutePath = path.resolve(filePath);
  const stats = await fs.lstat(absolutePath);

  if (!stats.isFile()) {
    throw new Error(`Expected a file path: ${absolutePath}`);
  }

  if (stats.isSymbolicLink()) {
    throw new Error(`Refusing symlink file: ${absolutePath}`);
  }

  await fs.access(absolutePath, constants.R_OK);
}

function buildRunConfigFromOptions(
  options: CliOptions,
  modelPath: string,
  outputDir: string,
  defaults?: { targetRatio?: number }
): {
  modelPath: string;
  outputDir: string;
  targetRatio?: number;
  calibrationRounds?: number;
  minExpertsPerLayer?: number;
  pruneMethod?: PruneMethod;
  nExpertsToPrunePerLayer?: number;
  preserveSuperExperts?: boolean;
  preserveOutliers?: boolean;
  allowLegacySaliency?: boolean;
  jobId?: string;
  observationPath?: string;
} {
  const ratioOption = optionString(options, 'ratio');
  let targetRatio =
    typeof ratioOption === 'string'
      ? assertFiniteNumber(ratioOption, 'ratio', 0, 0.95)
      : defaults?.targetRatio;

  const calibration = optionString(options, 'calibration');
  const calibrationRounds =
    typeof calibration === 'string'
      ? assertInteger(calibration, 'calibration', 1, 25)
      : undefined;
  const minExperts = optionString(options, 'min-experts');
  const minExpertsPerLayer =
    typeof minExperts === 'string'
      ? assertInteger(minExperts, 'min-experts', 1, 128)
      : undefined;

  const allowLegacySaliency = !optionBoolean(options, 'no-legacy');
  const pruneMethod = parsePruneMethod(optionString(options, 'prune-method'));
  const nExpertsToPrunePerLayerOption = optionString(options, 'n-experts-to-prune-per-layer');
  const nExpertsToPrunePerLayer =
    typeof nExpertsToPrunePerLayerOption === 'string'
      ? assertInteger(
          nExpertsToPrunePerLayerOption,
          'n-experts-to-prune-per-layer',
          0,
          10_000
        )
      : undefined;
  const preserveSuperExperts = optionBoolean(options, 'preserve-super-experts');
  const preserveOutliers = optionBoolean(options, 'preserve-outliers');

  if (preserveSuperExperts && preserveOutliers) {
    throw new Error('Only one of --preserve-super-experts or --preserve-outliers can be set');
  }

  if (typeof nExpertsToPrunePerLayer !== 'number' && typeof targetRatio !== 'number') {
    throw new Error('Missing required option --ratio unless --n-experts-to-prune-per-layer is set');
  }

  const jobId = optionString(options, 'job-id');
  const observationPath = optionString(options, 'observation');

  return {
    modelPath,
    outputDir,
    ...(typeof targetRatio === 'number' ? { targetRatio } : {}),
    ...(typeof calibrationRounds === 'number' ? { calibrationRounds } : {}),
    ...(typeof minExpertsPerLayer === 'number' ? { minExpertsPerLayer } : {}),
    ...(typeof pruneMethod === 'string' ? { pruneMethod } : {}),
    ...(typeof nExpertsToPrunePerLayer === 'number'
      ? { nExpertsToPrunePerLayer }
      : {}),
    ...(preserveSuperExperts ? { preserveSuperExperts: true } : {}),
    ...(preserveOutliers ? { preserveOutliers: true } : {}),
    ...(allowLegacySaliency ? {} : { allowLegacySaliency: false }),
    ...(jobId ? { jobId } : {}),
    ...(observationPath ? { observationPath } : {})
  };
}

async function handleRun(options: CliOptions): Promise<void> {
  const modelPath = requiredOption(options, 'model');
  const outputDir = requiredOption(options, 'output');

  await ensureFileReadable(modelPath);

  const runConfig = buildRunConfigFromOptions(options, modelPath, outputDir);
  const plan = await runReapMlx(runConfig);

  if (optionBoolean(options, 'json')) {
    process.stdout.write(`${JSON.stringify(plan, null, 2)}\n`);
    return;
  }

  printRunSummary(plan);
  const outputPlanPath = path.resolve(outputDir, 'pruning-plan.json');
  process.stdout.write(`plan written: ${outputPlanPath}\n`);
}

async function handleParity(options: CliOptions): Promise<void> {
  const leftModelPath = requiredOption(options, 'left');
  const rightModelPath = requiredOption(options, 'right');
  const outputDir = requiredOption(options, 'output');

  await ensureFileReadable(leftModelPath);
  await ensureFileReadable(rightModelPath);

  const runConfig = buildRunConfigFromOptions(options, leftModelPath, outputDir);
  const report = await runParityHarness({
    leftModelPath,
    rightModelPath,
    outputDir,
    ...(typeof runConfig.targetRatio === 'number' ? { targetRatio: runConfig.targetRatio } : {}),
    ...(typeof runConfig.calibrationRounds === 'number'
      ? { calibrationRounds: runConfig.calibrationRounds }
      : {}),
    ...(typeof runConfig.minExpertsPerLayer === 'number'
      ? { minExpertsPerLayer: runConfig.minExpertsPerLayer }
      : {}),
    ...(runConfig.pruneMethod ? { pruneMethod: runConfig.pruneMethod } : {}),
    ...(typeof runConfig.nExpertsToPrunePerLayer === 'number'
      ? { nExpertsToPrunePerLayer: runConfig.nExpertsToPrunePerLayer }
      : {}),
    ...(runConfig.preserveSuperExperts ? { preserveSuperExperts: true } : {}),
    ...(runConfig.preserveOutliers ? { preserveOutliers: true } : {}),
    ...(runConfig.allowLegacySaliency === false ? { allowLegacySaliency: false } : {}),
    ...(optionBoolean(options, 'require-identical-telemetry')
      ? { requireIdenticalTelemetry: true }
      : {})
  });

  if (optionBoolean(options, 'json')) {
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  } else {
    process.stdout.write(`left telemetry hash: ${report.leftTelemetry.normalizedHash}\n`);
    process.stdout.write(`right telemetry hash: ${report.rightTelemetry.normalizedHash}\n`);
    process.stdout.write(
      `telemetry exact match: ${report.telemetry.normalizedHashEqual ? 'yes' : 'no'}\n`
    );
    process.stdout.write(`pruned exact match: ${report.pruning.prunedExactMatch ? 'yes' : 'no'}\n`);
    process.stdout.write(`kept exact match: ${report.pruning.keptExactMatch ? 'yes' : 'no'}\n`);
    process.stdout.write(`threshold delta: ${report.pruning.thresholdDelta}\n`);
    process.stdout.write(`report written: ${report.artifacts.reportJsonPath}\n`);
    process.stdout.write(`markdown written: ${report.artifacts.reportMarkdownPath}\n`);
  }

  if (optionBoolean(options, 'require-identical-telemetry') && !report.telemetry.normalizedHashEqual) {
    throw new Error('Normalized telemetry mismatch between --left and --right');
  }

  if (!report.pruning.prunedExactMatch) {
    throw new Error('Pruned expert mismatch between --left and --right');
  }
}

async function handleCollect(options: CliOptions): Promise<void> {
  const modelPath = requiredOption(options, 'model');
  const outputDir = requiredOption(options, 'output');
  const collectConfig = buildCollectConfigFromOptions(options, {
    modelPath,
    outputDir
  });
  const result = await collectTelemetryWithMlx(collectConfig);

  process.stdout.write(`telemetry written: ${result.telemetryPath}\n`);
  process.stdout.write(`model: ${result.telemetry.modelName}\n`);
  process.stdout.write(`experts: ${result.telemetry.experts.length}\n`);
  printTelemetryCollectionSummary(result.telemetry.metadata);
}

async function runApplyWithConfig(config: {
  modelPath: string;
  planPath: string;
  outputDir: string;
  pythonBin?: string;
  dryRun?: boolean;
}) {
  await ensureFileReadable(config.planPath);

  return applyPruningPlanToMlxModel({
    modelPath: config.modelPath,
    planPath: config.planPath,
    outputDir: config.outputDir,
    ...(config.pythonBin ? { pythonBin: config.pythonBin } : {}),
    ...(config.dryRun ? { dryRun: true } : {})
  });
}

async function handleApply(options: CliOptions): Promise<void> {
  const modelPath = requiredOption(options, 'model');
  const planPath = requiredOption(options, 'plan');
  const outputDir = requiredOption(options, 'output');
  const pythonBin = optionString(options, 'python');
  const dryRun = optionBoolean(options, 'dry-run');

  const result = await runApplyWithConfig({
    modelPath,
    planPath,
    outputDir,
    ...(pythonBin ? { pythonBin } : {}),
    ...(dryRun ? { dryRun: true } : {})
  });

  process.stdout.write(`output model: ${result.outputModelPath}\n`);
  process.stdout.write(`output config: ${result.outputConfigPath}\n`);
  process.stdout.write(`layers patched: ${result.layersPatched}\n`);
  process.stdout.write(`experts before: ${result.expertsBefore}\n`);
  process.stdout.write(`experts after: ${result.expertsAfter}\n`);
  process.stdout.write(`plan job id: ${result.pruningPlanJobId}\n`);
}

async function handleProbe(options: CliOptions): Promise<void> {
  const modelPath = requiredOption(options, 'model');
  const prompt = requiredOption(options, 'prompt');
  const pythonBin = optionString(options, 'python');
  const maxTokens = assertInteger(optionString(options, 'max-tokens') ?? 80, 'max-tokens', 1, 4096);
  const temperature = assertFiniteNumber(optionString(options, 'temperature') ?? 0.2, 'temperature', 0, 5);

  const result = await probeMlxModelCoherence({
    modelPath,
    prompt,
    ...(pythonBin ? { pythonBin } : {}),
    maxTokens,
    temperature
  });

  process.stdout.write(`${result.completion.trim()}\n`);
}

async function handleFull(options: CliOptions): Promise<void> {
  const modelPath = requiredOption(options, 'model');
  const outputDir = requiredOption(options, 'output');
  const collectConfig = buildCollectConfigFromOptions(options, {
    modelPath,
    outputDir: path.resolve(outputDir, 'telemetry')
  });
  const pythonBin = collectConfig.pythonBin;
  const dryRun = optionBoolean(options, 'dry-run');

  await ensureSecureDir(outputDir);
  await ensureSecureDir(collectConfig.outputDir);
  const collectResult = await collectTelemetryWithMlx(collectConfig);

  const planOutputDir = path.resolve(outputDir, 'plan');
  await ensureSecureDir(planOutputDir);
  const runConfig = buildRunConfigFromOptions(options, collectResult.telemetryPath, planOutputDir, {
    targetRatio: 0.5
  });
  const plan = await runReapMlx(runConfig);

  const applyOutputDir = path.resolve(outputDir, 'pruned-model');
  await ensureSecureDir(applyOutputDir);
  const applyResult = await runApplyWithConfig({
    modelPath,
    planPath: path.resolve(planOutputDir, 'pruning-plan.json'),
    outputDir: applyOutputDir,
    ...(pythonBin ? { pythonBin } : {}),
    ...(dryRun ? { dryRun: true } : {})
  });

  const payload = {
    telemetryPath: collectResult.telemetryPath,
    telemetryExperts: collectResult.telemetry.experts.length,
    telemetryMetadata: collectResult.telemetry.metadata ?? {},
    planPath: path.resolve(planOutputDir, 'pruning-plan.json'),
    plan,
    apply: applyResult
  };

  if (optionBoolean(options, 'json')) {
    process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
    return;
  }

  process.stdout.write(`telemetry written: ${payload.telemetryPath}\n`);
  process.stdout.write(`plan written: ${payload.planPath}\n`);
  process.stdout.write(`output model: ${applyResult.outputModelPath}\n`);
  process.stdout.write(`output config: ${applyResult.outputConfigPath}\n`);
  process.stdout.write(`layers patched: ${applyResult.layersPatched}\n`);
  process.stdout.write(`experts before: ${applyResult.expertsBefore}\n`);
  process.stdout.write(`experts after: ${applyResult.expertsAfter}\n`);
  printTelemetryCollectionSummary(payload.telemetryMetadata);
}

async function handleObserve(options: CliOptions): Promise<void> {
  const observationFile = requiredOption(options, 'file');
  await ensureFileReadable(observationFile);

  const summary = await summarizeObservationLog(observationFile);

  if (optionBoolean(options, 'json')) {
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
    return;
  }

  printObservationSummary(summary);
}

async function handleInit(options: CliOptions): Promise<void> {
  const outputFile = requiredOption(options, 'output');
  const modelName = optionString(options, 'model-name') ?? 'synthetic-moe';

  const layers = assertInteger(optionString(options, 'layers') ?? 8, 'layers', 1, 512);
  const experts = assertInteger(optionString(options, 'experts') ?? 8, 'experts', 2, 512);
  const seedOption = optionString(options, 'seed');
  const seed =
    typeof seedOption === 'string'
      ? assertInteger(seedOption, 'seed', 1, 2147483647)
      : randomUUID()
          .replace(/-/g, '')
          .slice(0, 8)
          .split('')
          .reduce((acc: number, char: string) => acc + char.charCodeAt(0), 0);

  const telemetry = buildSyntheticTelemetry({
    modelName,
    layers,
    experts,
    seed
  });

  const absoluteOutput = path.resolve(outputFile);
  await ensureSecureDir(path.dirname(absoluteOutput));
  await writeJsonAtomicSafe(absoluteOutput, telemetry);

  process.stdout.write(`telemetry written: ${absoluteOutput}\n`);
  process.stdout.write(`experts: ${telemetry.experts.length}\n`);
}

async function resolveVersion(): Promise<string> {
  const packageJsonPath = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../package.json');
  const raw = await fs.readFile(packageJsonPath, 'utf8');
  const parsed = JSON.parse(raw) as { version?: string };
  return parsed.version ?? '0.0.0';
}

export async function main(argv = process.argv): Promise<void> {
  const parsed = parseArgs(argv);

  switch (parsed.command) {
    case 'run': {
      await handleRun(parsed.options);
      return;
    }
    case 'parity': {
      await handleParity(parsed.options);
      return;
    }
    case 'collect': {
      await handleCollect(parsed.options);
      return;
    }
    case 'full': {
      await handleFull(parsed.options);
      return;
    }
    case 'apply': {
      await handleApply(parsed.options);
      return;
    }
    case 'probe': {
      await handleProbe(parsed.options);
      return;
    }
    case 'observe': {
      await handleObserve(parsed.options);
      return;
    }
    case 'init': {
      await handleInit(parsed.options);
      return;
    }
    case 'version': {
      const version = await resolveVersion();
      process.stdout.write(`${version}\n`);
      return;
    }
    case 'help':
    default: {
      printUsage();
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`reap-mlx error: ${message}\n`);
    process.exitCode = 1;
  });
}
