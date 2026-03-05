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
import mlx.nn as nn
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
            flat_h = h.reshape(-1, h.shape[-1])
            num_tokens = int(flat_h.shape[0])

            gate = layer.mlp.gate
            switch = layer.mlp.switch_mlp
            num_experts = int(layer.mlp.num_experts)

            routing_weights = mx.softmax(gate(flat_h).astype(mx.float32), axis=-1)
            topk = int(layer.mlp.top_k)
            inds = mx.argpartition(-routing_weights, kth=topk - 1, axis=-1)[..., :topk]
            mx.eval(inds, routing_weights)

            if args.renorm_topk:
                topk_weights = mx.take_along_axis(routing_weights, inds, axis=-1)
                denom = mx.maximum(mx.sum(topk_weights, axis=-1, keepdims=True), 1e-12)
                routing_weights = routing_weights / denom

            topk_list = inds.tolist()
            token_sets = [[] for _ in range(num_experts)]
            for token_idx in range(num_tokens):
                for expert_id in topk_list[token_idx]:
                    token_sets[int(expert_id)].append(token_idx)

            for expert_id in range(num_experts):
                active_tokens = token_sets[expert_id]
                if len(active_tokens) == 0:
                    continue

                idx = mx.array(active_tokens)
                active_h = flat_h[idx]

                gp = switch.gate_proj
                up = switch.up_proj
                dp = switch.down_proj

                g_w = mx.dequantize(
                    gp.weight[expert_id:expert_id + 1],
                    gp.scales[expert_id:expert_id + 1],
                    gp.biases[expert_id:expert_id + 1],
                    gp.group_size,
                    gp.bits,
                ).squeeze(0)
                u_w = mx.dequantize(
                    up.weight[expert_id:expert_id + 1],
                    up.scales[expert_id:expert_id + 1],
                    up.biases[expert_id:expert_id + 1],
                    up.group_size,
                    up.bits,
                ).squeeze(0)
                d_w = mx.dequantize(
                    dp.weight[expert_id:expert_id + 1],
                    dp.scales[expert_id:expert_id + 1],
                    dp.biases[expert_id:expert_id + 1],
                    dp.group_size,
                    dp.bits,
                ).squeeze(0)

                expert_out = (nn.silu(active_h @ g_w.T) * (active_h @ u_w.T)) @ d_w.T
                ean_norm = mx.sqrt(mx.sum(expert_out * expert_out, axis=-1))
                active_rw = routing_weights[idx, expert_id]
                weighted_norm = ean_norm * active_rw
                mx.eval(ean_norm, active_rw, weighted_norm)

                key = (layer_id, expert_id)
                entry = stats.get(key)
                if entry is None:
                    entry = {
                        'activeTokenCount': 0,
                        'gateValueSum': 0.0,
                        'activationNormSum': 0.0,
                        'weightedActivationNormSum': 0.0,
                    }
                    stats[key] = entry

                entry['activeTokenCount'] += len(active_tokens)
                entry['gateValueSum'] += float(mx.sum(active_rw).item())
                entry['activationNormSum'] += float(mx.sum(ean_norm).item())
                entry['weightedActivationNormSum'] += float(mx.sum(weighted_norm).item())

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
