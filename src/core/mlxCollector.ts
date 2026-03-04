import { spawnSync, type SpawnSyncReturns } from 'node:child_process';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { assertInteger, ensureSecureDir, readJsonFileSafe, resolveSafePath } from './security.js';
import type { ModelTelemetry } from './types.js';

export interface MlxCollectConfig {
  modelPath: string;
  outputDir: string;
  prompt: string;
  maxTokens?: number;
  includeLayers?: string;
  renormTopK?: boolean;
  pythonBin?: string;
}

function pythonScript(): string {
  return `import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


def parse_layers(spec, max_layers):
    if spec is None or spec.strip() == "":
        return list(range(max_layers))

    result = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            start = int(a)
            end = int(b)
            step = 1 if start <= end else -1
            result.extend(list(range(start, end + step, step)))
        else:
            result.append(int(part))

    dedup = sorted(set(result))
    return [x for x in dedup if 0 <= x < max_layers]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--layers', default='')
    parser.add_argument('--renorm-topk', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    model, tokenizer = load(args.model, lazy=False)

    tokens = tokenizer.encode(args.prompt)
    if len(tokens) > args.max_tokens:
        tokens = tokens[:args.max_tokens]
    x = mx.array([tokens])

    layers = model.model.layers
    num_layers = len(layers)
    selected_layers = parse_layers(args.layers if args.layers else None, num_layers)

    stats = {}

    mask = None
    hidden = model.model.embed_tokens(x)

    if x.shape[1] > 1:
        mask = mx.full((x.shape[1], x.shape[1]), -1e9)
        mask = mx.triu(mask, k=1)
        mask = mask.astype(hidden.dtype)

    for layer_id, layer in enumerate(layers):
        if layer_id in selected_layers:
            h = layer.input_layernorm(hidden)

            gates = layer.mlp.gate(h)
            gates = mx.softmax(gates.astype(mx.float32), axis=-1, precise=True)

            topk = int(layer.mlp.top_k)
            inds = mx.argpartition(-gates, kth=topk - 1, axis=-1)[..., :topk]
            scores = mx.take_along_axis(gates, inds, axis=-1)

            if args.renorm_topk:
                denom = mx.maximum(mx.sum(scores, axis=-1, keepdims=True), 1e-12)
                scores = scores / denom

            expert_out = layer.mlp.switch_mlp(h, inds)
            expert_norm = mx.sqrt(
                mx.sum(expert_out.astype(mx.float32) * expert_out.astype(mx.float32), axis=-1)
            )

            flat_inds = inds.reshape(-1).tolist()
            flat_scores = scores.reshape(-1).tolist()
            flat_norms = expert_norm.reshape(-1).tolist()

            for expert_id, gate_val, norm_val in zip(flat_inds, flat_scores, flat_norms):
                key = (layer_id, int(expert_id))
                entry = stats.get(key)
                if entry is None:
                    entry = {
                        'activeTokenCount': 0,
                        'gateValueSum': 0.0,
                        'activationNormSum': 0.0,
                        'weightedActivationNormSum': 0.0,
                    }
                    stats[key] = entry

                entry['activeTokenCount'] += 1
                entry['gateValueSum'] += float(gate_val)
                entry['activationNormSum'] += float(norm_val)
                entry['weightedActivationNormSum'] += float(gate_val) * float(norm_val)

        hidden = layer(hidden, mask=mask)

    experts = []
    num_experts_by_layer = {}

    for layer_id in selected_layers:
        layer_num_experts = int(layers[layer_id].mlp.num_experts)
        num_experts_by_layer[layer_id] = layer_num_experts

        for expert_id in range(layer_num_experts):
            entry = stats.get(
                (layer_id, expert_id),
                {
                    'activeTokenCount': 0,
                    'gateValueSum': 0.0,
                    'activationNormSum': 0.0,
                    'weightedActivationNormSum': 0.0,
                }
            )

            active = int(entry['activeTokenCount'])
            denom = max(1, active)
            experts.append({
                'layer': layer_id,
                'expert': expert_id,
                'activeTokenCount': active,
                'tokenCount': active,
                'gateValueSum': round(float(entry['gateValueSum']), 8),
                'activationNormSum': round(float(entry['activationNormSum']), 8),
                'weightedActivationNormSum': round(float(entry['weightedActivationNormSum']), 8),
                'averageGateValue': round(float(entry['gateValueSum']) / denom, 8),
                'averageActivationNorm': round(float(entry['activationNormSum']) / denom, 8),
                'activationScore': round(float(entry['weightedActivationNormSum']) / denom, 8),
            })

    payload = {
        'modelName': Path(args.model).name,
        'experts': experts,
        'metadata': {
            'source': 'mlx_real_collector_exact',
            'promptLengthTokens': int(len(tokens)),
            'selectedLayers': ','.join(str(x) for x in selected_layers),
            'topK': int(layers[0].mlp.top_k if len(layers) > 0 else 0),
            'numExpertsByLayer': {str(k): int(v) for k, v in num_experts_by_layer.items()},
            'renormTopK': bool(args.renorm_topk),
        }
    }

    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
        f.write('\\n')

    print(args.output)

if __name__ == '__main__':
    main()
`;
}

function buildArgs(config: MlxCollectConfig, telemetryPath: string): string[] {
  const maxTokens = assertInteger(config.maxTokens ?? 256, 'maxTokens', 1, 8192);

  return [
    '-c',
    pythonScript(),
    '--model',
    path.resolve(config.modelPath),
    '--prompt',
    config.prompt,
    '--max-tokens',
    String(maxTokens),
    '--layers',
    config.includeLayers ?? '',
    ...(config.renormTopK ? ['--renorm-topk'] : []),
    '--output',
    telemetryPath
  ];
}

function throwIfFailed(result: SpawnSyncReturns<string>, commandPreview: string): void {
  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    const stderr = result.stderr?.trim() ?? '';
    const stdout = result.stdout?.trim() ?? '';
    throw new Error(
      `MLX telemetry collection failed (exit=${result.status})\n${commandPreview}\n${stderr || stdout}`
    );
  }
}

export async function collectTelemetryWithMlx(
  config: MlxCollectConfig
): Promise<{ telemetryPath: string; telemetry: ModelTelemetry }> {
  const outputDir = path.resolve(config.outputDir);
  await ensureSecureDir(outputDir);

  const telemetryPath = resolveSafePath(
    outputDir,
    `telemetry-${randomUUID()}.json`
  );

  const pythonBin = config.pythonBin && config.pythonBin.trim().length > 0
    ? config.pythonBin
    : 'python3';

  const args = buildArgs(config, telemetryPath);

  const commandPreview = `${pythonBin} ${args.join(' ')}`;
  const result = spawnSync(pythonBin, args, {
    encoding: 'utf8',
    maxBuffer: 20 * 1024 * 1024
  });

  throwIfFailed(result, commandPreview);

  const telemetry = await readJsonFileSafe<ModelTelemetry>(telemetryPath);

  return {
    telemetryPath,
    telemetry
  };
}
