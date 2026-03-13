import { spawnSync, type SpawnSyncReturns } from 'node:child_process';
import path from 'node:path';
import {
  assertFiniteNumber,
  ensureSecureDir,
  readJsonFileSafe,
  resolveSafePath
} from './security.js';
import type { PruningPlan } from './types.js';

export interface MlxApplyPlanConfig {
  modelPath: string;
  outputDir: string;
  planPath: string;
  pythonBin?: string;
  dryRun?: boolean;
}

export interface MlxApplyPlanResult {
  outputModelPath: string;
  outputConfigPath: string;
  layersPatched: number;
  expertsBefore: number;
  expertsAfter: number;
  pruningPlanJobId: string;
}

function pythonScript(): string {
  return `import argparse
import json
from pathlib import Path
import shutil

import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import save_model


def resolve_layers(model):
    language_model = getattr(model, 'language_model', None)
    if language_model is not None:
        nested_model = getattr(language_model, 'model', None)
        nested_layers = getattr(nested_model, 'layers', None) if nested_model is not None else None
        if nested_layers is not None:
            return nested_layers

    nested_model = getattr(model, 'model', None)
    nested_layers = getattr(nested_model, 'layers', None) if nested_model is not None else None
    if nested_layers is not None:
        return nested_layers

    top_level_layers = getattr(model, 'layers', None)
    if top_level_layers is not None:
        return top_level_layers

    raise RuntimeError(
        f'Unable to resolve transformer layers from model type={type(model).__name__}'
    )


def parse_keep_map(plan):
    keep_map = {}
    kept = plan.get('kept', [])
    for item in kept:
        layer = int(item['layer'])
        expert = int(item['expert'])
        keep_map.setdefault(layer, set()).add(expert)

    # if kept is empty, infer from pruned by leaving untouched is impossible;
    # require explicit kept list for safe apply.
    if not keep_map:
        raise RuntimeError('Plan missing kept experts; cannot apply structurally')

    return {layer: sorted(experts) for layer, experts in keep_map.items()}


def slice_expert_axis(module, keep):
    idx = mx.array(keep)

    if hasattr(module, 'weight'):
        module.weight = module.weight[idx]
    if hasattr(module, 'scales'):
        module.scales = module.scales[idx]
    if hasattr(module, 'biases') and module.biases is not None:
        module.biases = module.biases[idx]
    if hasattr(module, 'bias') and getattr(module, 'bias') is not None:
        module.bias = module.bias[idx]


def patch_layer(layer, keep):
    mlp = layer.mlp
    if not hasattr(mlp, 'gate') or not hasattr(mlp, 'switch_mlp'):
        raise RuntimeError('Unsupported layer MLP structure for structural pruning')

    slice_expert_axis(mlp.gate, keep)
    for name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(mlp.switch_mlp, name)
        slice_expert_axis(proj, keep)

    if hasattr(mlp, 'num_experts'):
        mlp.num_experts = len(keep)
    if hasattr(mlp, 'top_k'):
        mlp.top_k = min(int(mlp.top_k), len(keep))


def copy_non_weight_files(src: Path, dst: Path):
    for item in src.iterdir():
        if not item.is_file():
            continue
        if item.suffix == '.safetensors':
            continue
        if item.name == 'config.json':
            continue
        shutil.copy2(str(item), str(dst / item.name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    plan_path = Path(args.plan).resolve()

    plan = json.loads(plan_path.read_text())
    keep_map = parse_keep_map(plan)

    model, tokenizer = load(str(model_path), lazy=False)

    layers = resolve_layers(model)
    num_layers = len(layers)

    patched_layers = []
    experts_before = None
    experts_after = None

    for layer_idx in sorted(keep_map.keys()):
        if layer_idx < 0 or layer_idx >= num_layers:
            raise RuntimeError(f'Plan layer index out of range: {layer_idx}')

        keep = keep_map[layer_idx]
        if len(keep) == 0:
            raise RuntimeError(f'Plan keeps zero experts in layer {layer_idx}')

        layer = layers[layer_idx]
        before_count = int(layer.mlp.num_experts) if hasattr(layer.mlp, 'num_experts') else len(keep)

        for expert in keep:
            if expert < 0 or expert >= before_count:
                raise RuntimeError(f'Invalid expert index {expert} for layer {layer_idx}')

        patch_layer(layer, keep)

        after_count = len(keep)
        patched_layers.append(layer_idx)

        if experts_before is None:
            experts_before = before_count
        if experts_after is None:
            experts_after = after_count

    if experts_before is None:
        raise RuntimeError('No layers patched from plan')

    if args.dry_run:
        result = {
            'outputModelPath': str(output_path),
            'outputConfigPath': str(output_path / 'config.json'),
            'layersPatched': len(patched_layers),
            'expertsBefore': experts_before,
            'expertsAfter': experts_after,
            'pruningPlanJobId': plan.get('jobId', 'unknown')
        }
        print(json.dumps(result))
        return

    output_path.mkdir(parents=True, exist_ok=True)
    copy_non_weight_files(model_path, output_path)

    source_config = json.loads((model_path / 'config.json').read_text())

    min_keep = min(len(v) for v in keep_map.values())
    source_config['num_experts'] = min_keep
    if 'num_experts_per_tok' in source_config:
        source_config['num_experts_per_tok'] = min(
            int(source_config['num_experts_per_tok']),
            min_keep,
        )

    source_config['reap_mlx_pruning'] = {
        'method': 'reap',
        'planJobId': plan.get('jobId'),
        'sourcePlanPath': str(plan_path),
        'layersPatched': patched_layers,
        'expertsBefore': experts_before,
        'expertsAfter': experts_after,
    }

    (output_path / 'config.json').write_text(json.dumps(source_config, indent=2) + '\\n')

    mx.eval(model.parameters())
    save_model(output_path, model, donate_model=True)

    result = {
        'outputModelPath': str(output_path),
        'outputConfigPath': str(output_path / 'config.json'),
        'layersPatched': len(patched_layers),
        'expertsBefore': experts_before,
        'expertsAfter': experts_after,
        'pruningPlanJobId': plan.get('jobId', 'unknown')
    }

    print(json.dumps(result))


if __name__ == '__main__':
    main()
`;
}

function throwIfFailed(result: SpawnSyncReturns<string>, commandPreview: string): void {
  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    const stderr = result.stderr?.trim() ?? '';
    const stdout = result.stdout?.trim() ?? '';
    throw new Error(
      `MLX structural pruning failed (exit=${result.status})\n${commandPreview}\n${stderr || stdout}`
    );
  }
}

function parseResult(stdout: string): MlxApplyPlanResult {
  const trimmed = stdout.trim();
  const line = trimmed.split(/\r?\n/).at(-1);
  if (!line) {
    throw new Error('MLX structural pruning produced no JSON result');
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid JSON result from MLX structural pruning: ${message}`);
  }

  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Unexpected result payload from MLX structural pruning');
  }

  const payload = parsed as Record<string, unknown>;

  return {
    outputModelPath: String(payload.outputModelPath ?? ''),
    outputConfigPath: String(payload.outputConfigPath ?? ''),
    layersPatched: assertFiniteNumber(payload.layersPatched, 'layersPatched', 0),
    expertsBefore: assertFiniteNumber(payload.expertsBefore, 'expertsBefore', 0),
    expertsAfter: assertFiniteNumber(payload.expertsAfter, 'expertsAfter', 0),
    pruningPlanJobId: String(payload.pruningPlanJobId ?? 'unknown')
  };
}

export const __testOnly = {
  pythonScript
};

export async function applyPruningPlanToMlxModel(
  config: MlxApplyPlanConfig
): Promise<MlxApplyPlanResult> {
  const outputDir = path.resolve(config.outputDir);
  await ensureSecureDir(outputDir);

  const planPath = path.resolve(config.planPath);
  await readJsonFileSafe<PruningPlan>(planPath);

  const pythonBin = config.pythonBin && config.pythonBin.trim().length > 0
    ? config.pythonBin
    : 'python3';

  const args = [
    '-c',
    pythonScript(),
    '--model',
    path.resolve(config.modelPath),
    '--plan',
    planPath,
    '--output',
    outputDir,
    ...(config.dryRun ? ['--dry-run'] : [])
  ];

  const commandPreview = `${pythonBin} ${args.join(' ')}`;
  const result = spawnSync(pythonBin, args, {
    encoding: 'utf8',
    maxBuffer: 50 * 1024 * 1024
  });

  throwIfFailed(result, commandPreview);
  return parseResult(result.stdout ?? '');
}

export async function deriveOutputPlanPath(outputDir: string): Promise<string> {
  const pathValue = resolveSafePath(outputDir, 'pruning-plan.json');
  await readJsonFileSafe<PruningPlan>(pathValue);
  return pathValue;
}
