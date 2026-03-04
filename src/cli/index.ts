#!/usr/bin/env node

import { randomUUID } from 'node:crypto';
import { constants, promises as fs } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  applyPruningPlanToMlxModel,
  collectTelemetryWithMlx,
  runReapMlx,
  summarizeObservationLog
} from '../core/index.js';
import {
  assertFiniteNumber,
  assertInteger,
  ensureSecureDir,
  writeJsonAtomicSafe
} from '../core/security.js';
import type { ModelTelemetry, ObservationSummary, PruningPlan } from '../core/types.js';

type CliOptions = Map<string, string | boolean>;

interface ParsedCli {
  command: string;
  options: CliOptions;
  positionals: string[];
}

function printUsage(): void {
  process.stdout.write(`reap-mlx - secure REAP pruning planner for MLX telemetry

Usage:
  reap-mlx <command> [options]

Commands:
  run       Build pruning plan from telemetry JSON
  collect   Collect REAP telemetry from a real MLX model
  apply     Apply pruning plan to an MLX checkpoint
  observe   Summarize observation log
  init      Generate synthetic telemetry JSON
  help      Show this help
  version   Print version

run options:
  --model <file>           Telemetry JSON input file
  --output <dir>           Output directory for plan + logs
  --ratio <0..0.95>        Target prune ratio
  --calibration <1..25>    Calibration rounds (default: 2)
  --min-experts <1..128>   Minimum experts preserved per layer (default: 1)
  --no-legacy              Require REAP saliency fields (disable activationScore fallback)
  --job-id <id>            Optional job id
  --observation <name>     Observation filename (default: observation.log)
  --json                   Output full result as JSON

collect options:
  --model <dir>            Local MLX model directory
  --output <dir>           Output directory for telemetry JSON
  --prompt <text>          Prompt used to collect routing telemetry
  --max-tokens <1..8192>   Prompt token cap (default: 256)
  --layers <spec>          Optional layer filter (e.g. 0-3,8,10)
  --renorm-topk            Renormalize top-k gate weights to sum to 1
  --python <bin>           Python binary (default: python3)

apply options:
  --model <dir>            Source MLX model directory
  --plan <file>            Pruning plan JSON from reap-mlx run
  --output <dir>           Output pruned model directory
  --python <bin>           Python binary (default: python3)
  --dry-run                Validate and simulate patch without writing model

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

async function handleRun(options: CliOptions): Promise<void> {
  const modelPath = requiredOption(options, 'model');
  const outputDir = requiredOption(options, 'output');
  const targetRatio = assertFiniteNumber(requiredOption(options, 'ratio'), 'ratio', 0, 0.95);
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

  const jobId = optionString(options, 'job-id');
  const observationPath = optionString(options, 'observation');

  await ensureFileReadable(modelPath);

  const runConfig = {
    modelPath,
    outputDir,
    targetRatio,
    ...(typeof calibrationRounds === 'number' ? { calibrationRounds } : {}),
    ...(typeof minExpertsPerLayer === 'number' ? { minExpertsPerLayer } : {}),
    ...(allowLegacySaliency ? {} : { allowLegacySaliency: false }),
    ...(jobId ? { jobId } : {}),
    ...(observationPath ? { observationPath } : {})
  };

  const plan = await runReapMlx(runConfig);

  if (optionBoolean(options, 'json')) {
    process.stdout.write(`${JSON.stringify(plan, null, 2)}\n`);
    return;
  }

  printRunSummary(plan);
  const outputPlanPath = path.resolve(outputDir, 'pruning-plan.json');
  process.stdout.write(`plan written: ${outputPlanPath}\n`);
}

async function handleCollect(options: CliOptions): Promise<void> {
  const modelPath = requiredOption(options, 'model');
  const outputDir = requiredOption(options, 'output');
  const prompt = requiredOption(options, 'prompt');
  const maxTokens = assertInteger(
    optionString(options, 'max-tokens') ?? 256,
    'max-tokens',
    1,
    8192
  );
  const includeLayers = optionString(options, 'layers');
  const renormTopK = optionBoolean(options, 'renorm-topk');
  const pythonBin = optionString(options, 'python');

  const result = await collectTelemetryWithMlx({
    modelPath,
    outputDir,
    prompt,
    maxTokens,
    ...(includeLayers ? { includeLayers } : {}),
    ...(renormTopK ? { renormTopK: true } : {}),
    ...(pythonBin ? { pythonBin } : {})
  });

  process.stdout.write(`telemetry written: ${result.telemetryPath}\n`);
  process.stdout.write(`model: ${result.telemetry.modelName}\n`);
  process.stdout.write(`experts: ${result.telemetry.experts.length}\n`);
}

async function handleApply(options: CliOptions): Promise<void> {
  const modelPath = requiredOption(options, 'model');
  const planPath = requiredOption(options, 'plan');
  const outputDir = requiredOption(options, 'output');
  const pythonBin = optionString(options, 'python');
  const dryRun = optionBoolean(options, 'dry-run');

  await ensureFileReadable(planPath);

  const result = await applyPruningPlanToMlxModel({
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
          .reduce((acc, char) => acc + char.charCodeAt(0), 0);

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
    case 'collect': {
      await handleCollect(parsed.options);
      return;
    }
    case 'apply': {
      await handleApply(parsed.options);
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
