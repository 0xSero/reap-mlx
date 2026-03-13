import { spawnSync, type SpawnSyncReturns } from 'node:child_process';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import {
  assertInteger,
  ensureSecureDir,
  readJsonFileSafe,
  resolveSafePath,
  writeJsonAtomicSafe
} from './security.js';
import type { ModelTelemetry } from './types.js';

export type DatasetFormat = 'auto' | 'json' | 'jsonl' | 'csv' | 'parquet' | 'text';
export type CollectMode = 'single_pass' | 'replay_per_layer' | 'reload_per_layer';

export interface MlxCollectConfig {
  modelPath: string;
  outputDir: string;
  prompt?: string;
  datasetName?: string;
  datasetFile?: string;
  datasetFormat?: DatasetFormat;
  datasetSplit?: string;
  datasetTextField?: string;
  datasetMessagesField?: string;
  maxSamples?: number;
  minSamples?: number;
  maxTokens?: number;
  includeLayers?: string;
  renormTopK?: boolean;
  layerWise?: boolean;
  collectMode?: CollectMode;
  batchSize?: number;
  sampleBatchSize?: number;
  packSamples?: boolean;
  lazyLoad?: boolean;
  pythonBin?: string;
}

function pythonScript(): string {
  return `import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

NEWLINE = chr(10)
DOUBLE_NEWLINE = NEWLINE * 2


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


def normalize_collect_mode(args):
    if args.collect_mode is not None and args.collect_mode.strip() != '':
        mode = str(args.collect_mode).strip()
    elif args.layer_wise:
        mode = 'replay_per_layer'
    else:
        mode = 'single_pass'

    allowed = {'single_pass', 'replay_per_layer', 'reload_per_layer'}
    if mode not in allowed:
        raise RuntimeError(f'Unsupported collect mode: {mode}')
    if mode == 'reload_per_layer':
        return 'replay_per_layer'
    return mode


def normalize_dataset_format(dataset_file, format_hint):
    if format_hint is not None and format_hint != 'auto':
        return str(format_hint)

    suffix = Path(dataset_file).suffix.lower()
    if suffix == '.jsonl':
        return 'jsonl'
    if suffix == '.json':
        return 'json'
    if suffix == '.csv':
        return 'csv'
    if suffix == '.parquet':
        return 'parquet'
    if suffix == '.txt':
        return 'text'
    return 'jsonl'


def resolve_field(sample, field_path):
    if field_path is None or str(field_path).strip() == '':
        return None

    current = sample
    for segment in str(field_path).split('.'):
        if isinstance(current, dict):
            current = current.get(segment)
            continue
        if isinstance(current, list) and segment.isdigit():
            idx = int(segment)
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None

    return current


def iter_dataset_rows(args):
    split = args.dataset_split if args.dataset_split else 'train'

    if args.dataset_file is not None and args.dataset_file.strip() != '':
        dataset_file = str(Path(args.dataset_file).resolve())
        dataset_format = normalize_dataset_format(dataset_file, args.dataset_format)

        if dataset_format == 'text':
            with open(dataset_file, 'r', encoding='utf8') as handle:
                for line in handle:
                    text = line.rstrip(NEWLINE)
                    yield {'text': text, 'instruction': text}
            return

        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError('datasets package is required for --dataset-file mode') from e

        loader_name = 'json' if dataset_format in {'json', 'jsonl'} else dataset_format
        ds = load_dataset(loader_name, data_files=dataset_file, split=split)
        for sample in ds:
            yield sample
        return

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError('datasets package is required for --dataset mode') from e

    ds = load_dataset(args.dataset, split=split)
    for sample in ds:
        yield sample


def render_messages(messages, tokenizer):
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return None

    if not isinstance(messages, list):
        return None

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass
    except Exception:
        pass

    rendered = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get('role', 'user'))
        content = str(item.get('content', '')).strip()
        if content == '':
            continue
        rendered.append(f"{role}: {content}")

    if len(rendered) == 0:
        return None

    return DOUBLE_NEWLINE.join(rendered)


def fallback_text(sample):
    for field in ['instruction', 'text', 'content', 'prompt', 'question']:
        value = resolve_field(sample, field)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() != '':
            return value.strip()
        if isinstance(value, list):
            joined = NEWLINE.join(str(part) for part in value if str(part).strip() != '')
            if joined.strip() != '':
                return joined.strip()
    return None


def extract_sample_text(sample, args, tokenizer):
    if args.dataset_messages_field is not None and args.dataset_messages_field.strip() != '':
        rendered = render_messages(resolve_field(sample, args.dataset_messages_field), tokenizer)
        if rendered is None:
            return None
        return str(rendered).strip()

    raw_text = None
    if args.dataset_text_field is not None and args.dataset_text_field.strip() != '':
        raw_text = resolve_field(sample, args.dataset_text_field)
    else:
        raw_text = fallback_text(sample)

    if raw_text is None:
        return None

    if isinstance(raw_text, list):
        text = NEWLINE.join(str(part) for part in raw_text)
    elif isinstance(raw_text, dict):
        text = json.dumps(raw_text, ensure_ascii=False)
    else:
        text = str(raw_text)

    text = text.strip()
    if text == '':
        return None

    return text


def get_separator_token_ids(tokenizer):
    if getattr(tokenizer, 'eos_token_id', None) is not None:
        return [int(tokenizer.eos_token_id)]
    return tokenizer.encode(DOUBLE_NEWLINE)


def get_pad_token_id(tokenizer):
    if getattr(tokenizer, 'pad_token_id', None) is not None:
        return int(tokenizer.pad_token_id)
    if getattr(tokenizer, 'eos_token_id', None) is not None:
        return int(tokenizer.eos_token_id)
    return 0


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


def resolve_embed_tokens(model):
    language_model = getattr(model, 'language_model', None)
    if language_model is not None:
        nested_model = getattr(language_model, 'model', None)
        embed_tokens = getattr(nested_model, 'embed_tokens', None) if nested_model is not None else None
        if embed_tokens is not None:
            return embed_tokens

    nested_model = getattr(model, 'model', None)
    embed_tokens = getattr(nested_model, 'embed_tokens', None) if nested_model is not None else None
    if embed_tokens is not None:
        return embed_tokens

    top_level_embed_tokens = getattr(model, 'embed_tokens', None)
    if top_level_embed_tokens is not None:
        return top_level_embed_tokens

    raise RuntimeError(
        f'Unable to resolve text embedding weights from model type={type(model).__name__}'
    )


def iter_token_samples(args, tokenizer, counters):
    max_samples = int(args.max_samples) if args.max_samples is not None else 100
    min_samples = int(args.min_samples) if args.min_samples is not None else 1

    if args.prompt is not None and args.prompt.strip() != '':
        counters['scannedSamples'] += 1
        tokens = tokenizer.encode(args.prompt)
        if len(tokens) > args.max_tokens:
            tokens = tokens[:args.max_tokens]
        if len(tokens) == 0:
            raise RuntimeError('Prompt produced zero tokens after tokenization')
        counters['processedSamples'] += 1
        counters['sourceTokenCount'] += len(tokens)
        yield tokens
        return

    produced_any = False
    for sample in iter_dataset_rows(args):
        counters['scannedSamples'] += 1
        if counters['processedSamples'] >= max_samples:
            break

        text = extract_sample_text(sample, args, tokenizer)
        if text is None:
            counters['skippedSamples'] += 1
            continue

        tokens = tokenizer.encode(text)
        if len(tokens) > args.max_tokens:
            tokens = tokens[:args.max_tokens]
        if len(tokens) == 0:
            counters['skippedSamples'] += 1
            continue

        counters['processedSamples'] += 1
        counters['sourceTokenCount'] += len(tokens)
        produced_any = True
        yield tokens

    if counters['processedSamples'] < min_samples:
        raise RuntimeError(
            f'Processed {counters["processedSamples"]} samples but min-samples={min_samples} was requested'
        )

    if not produced_any:
        source_label = args.dataset if args.dataset is not None else args.dataset_file
        raise RuntimeError(f'No usable samples found for dataset source={source_label}')


def iter_sequences(args, tokenizer, counters):
    sample_iter = iter_token_samples(args, tokenizer, counters)

    if not args.pack_samples:
        for tokens in sample_iter:
            counters['modelTokenCount'] += len(tokens)
            counters['packedSequences'] += 1
            yield tokens
        return

    separator_tokens = get_separator_token_ids(tokenizer)
    current = []

    for tokens in sample_iter:
        if len(current) == 0:
            candidate = list(tokens)
        else:
            candidate = current + separator_tokens + list(tokens)

        if len(candidate) > args.max_tokens:
            if len(current) > 0:
                counters['modelTokenCount'] += len(current)
                counters['packedSequences'] += 1
                yield current
                current = list(tokens)
            else:
                current = list(tokens[:args.max_tokens])

            if len(current) >= args.max_tokens:
                counters['modelTokenCount'] += len(current)
                counters['packedSequences'] += 1
                yield current
                current = []
            continue

        current = candidate

        if len(current) >= args.max_tokens:
            counters['modelTokenCount'] += len(current)
            counters['packedSequences'] += 1
            yield current
            current = []

    if len(current) > 0:
        counters['modelTokenCount'] += len(current)
        counters['packedSequences'] += 1
        yield current


def iter_sequence_batches(args, tokenizer, counters):
    sample_batch_size = int(args.sample_batch_size) if args.sample_batch_size is not None else 1
    current_batch = []

    for sequence in iter_sequences(args, tokenizer, counters):
        current_batch.append(sequence)
        if len(current_batch) >= sample_batch_size:
            counters['sampleBatches'] += 1
            yield current_batch
            current_batch = []

    if len(current_batch) > 0:
        counters['sampleBatches'] += 1
        yield current_batch


def pad_batch(sequences, pad_token_id):
    max_len = max(len(sequence) for sequence in sequences)
    padded = []
    valid = []
    for sequence in sequences:
        pad = max_len - len(sequence)
        padded.append(sequence + ([pad_token_id] * pad))
        valid.append(([1] * len(sequence)) + ([0] * pad))
    return padded, valid


def expert_weight(module, expert_id):
    if (
        hasattr(module, 'scales')
        and hasattr(module, 'biases')
        and hasattr(module, 'group_size')
        and hasattr(module, 'bits')
    ):
        return mx.dequantize(
            module.weight[expert_id:expert_id + 1],
            module.scales[expert_id:expert_id + 1],
            module.biases[expert_id:expert_id + 1],
            module.group_size,
            module.bits,
        ).squeeze(0).astype(mx.float32)

    return module.weight[expert_id].astype(mx.float32)


def update_stats_for_hidden(layer_id, layer, hidden, valid_mask, stats, renorm_topk, batch_size=None):
    h = layer.input_layernorm(hidden)
    flat_h = h.reshape(-1, h.shape[-1])

    if valid_mask is not None:
        flat_valid = valid_mask.reshape(-1).tolist()
        valid_indices = [idx for idx, flag in enumerate(flat_valid) if int(flag) == 1]
        if len(valid_indices) == 0:
            return
        flat_h = flat_h[mx.array(valid_indices)]

    total_tokens = int(flat_h.shape[0])
    if total_tokens == 0:
        return

    gate = layer.mlp.gate
    switch = layer.mlp.switch_mlp
    num_experts = int(layer.mlp.num_experts)
    topk = min(int(layer.mlp.top_k), num_experts)

    if batch_size is None or batch_size >= total_tokens:
        ranges = [(0, total_tokens)]
    else:
        ranges = []
        start = 0
        while start < total_tokens:
            stop = min(start + batch_size, total_tokens)
            ranges.append((start, stop))
            start = stop

    for start, stop in ranges:
        chunk_h = flat_h[start:stop]
        num_tokens = int(chunk_h.shape[0])

        routing_weights = mx.softmax(gate(chunk_h).astype(mx.float32), axis=-1)
        inds = mx.argpartition(-routing_weights, kth=topk - 1, axis=-1)[..., :topk]
        mx.eval(inds, routing_weights)

        if renorm_topk:
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
            active_h = chunk_h[idx].astype(mx.float32)

            gp = switch.gate_proj
            up = switch.up_proj
            dp = switch.down_proj

            g_w = expert_weight(gp, expert_id)
            u_w = expert_weight(up, expert_id)
            d_w = expert_weight(dp, expert_id)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt')
    parser.add_argument('--dataset')
    parser.add_argument('--dataset-file')
    parser.add_argument('--dataset-format', default='auto')
    parser.add_argument('--dataset-split', default='train')
    parser.add_argument('--dataset-text-field', default='')
    parser.add_argument('--dataset-messages-field', default='')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--min-samples', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--sample-batch-size', type=int, default=1)
    parser.add_argument('--layers', default='')
    parser.add_argument('--renorm-topk', action='store_true')
    parser.add_argument('--layer-wise', action='store_true')
    parser.add_argument('--collect-mode', default='single_pass')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--pack-samples', action='store_true')
    parser.add_argument('--lazy-load', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    sources = 0
    if args.prompt is not None and args.prompt.strip() != '':
        sources += 1
    if args.dataset is not None and args.dataset.strip() != '':
        sources += 1
    if args.dataset_file is not None and args.dataset_file.strip() != '':
        sources += 1

    if sources != 1:
        raise RuntimeError('Exactly one of --prompt, --dataset, or --dataset-file is required')
    if int(args.max_samples) < 1:
        raise RuntimeError('max-samples must be >= 1')
    if args.batch_size is not None and int(args.batch_size) < 1:
        raise RuntimeError('batch-size must be >= 1')
    if int(args.sample_batch_size) < 1:
        raise RuntimeError('sample-batch-size must be >= 1')
    if int(args.min_samples) < 1:
        raise RuntimeError('min-samples must be >= 1')
    if int(args.min_samples) > int(args.max_samples):
        raise RuntimeError('min-samples cannot exceed max-samples')
    if int(args.max_tokens) < 1:
        raise RuntimeError('max-tokens must be >= 1')

    collect_mode = normalize_collect_mode(args)

    model, tokenizer = load(args.model, lazy=bool(args.lazy_load))

    if args.prompt is not None and args.prompt.strip() != '':
        input_mode = 'prompt'
    elif args.dataset_file is not None and args.dataset_file.strip() != '':
        input_mode = 'dataset_file'
    else:
        input_mode = 'dataset'

    layers = resolve_layers(model)
    num_layers = len(layers)
    selected_layers = parse_layers(args.layers if args.layers else None, num_layers)
    selected_layer_set = set(selected_layers)

    stats = {}
    counters = {
        'scannedSamples': 0,
        'processedSamples': 0,
        'skippedSamples': 0,
        'sourceTokenCount': 0,
        'modelTokenCount': 0,
        'packedSequences': 0,
        'sampleBatches': 0,
    }

    pad_token_id = get_pad_token_id(tokenizer)
    embed_tokens = resolve_embed_tokens(model)

    for sequences in iter_sequence_batches(args, tokenizer, counters):
        padded, valid = pad_batch(sequences, pad_token_id)
        x = mx.array(padded)
        valid_mask = mx.array(valid)

        hidden = embed_tokens(x)

        mask = None
        if x.shape[1] > 1:
            mask = mx.full((x.shape[1], x.shape[1]), -1e9)
            mask = mx.triu(mask, k=1)
            mask = mask.astype(hidden.dtype)

        if collect_mode == 'replay_per_layer':
            for target_layer in selected_layers:
                local_hidden = hidden
                for layer_id, layer in enumerate(layers):
                    if layer_id == target_layer:
                        update_stats_for_hidden(
                            target_layer,
                            layer,
                            local_hidden,
                            valid_mask,
                            stats,
                            args.renorm_topk,
                            args.batch_size,
                        )
                        break
                    local_hidden = layer(local_hidden, mask=mask)
        else:
            local_hidden = hidden
            for layer_id, layer in enumerate(layers):
                if layer_id in selected_layer_set:
                    update_stats_for_hidden(
                        layer_id,
                        layer,
                        local_hidden,
                        valid_mask,
                        stats,
                        args.renorm_topk,
                        args.batch_size,
                    )

                local_hidden = layer(local_hidden, mask=mask)

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
            'inputMode': input_mode,
            'scannedSamples': int(counters['scannedSamples']),
            'processedSamples': int(counters['processedSamples']),
            'skippedSamples': int(counters['skippedSamples']),
            'promptLengthTokens': int(counters['modelTokenCount']),
            'modelTokenCount': int(counters['modelTokenCount']),
            'sourceTokenCount': int(counters['sourceTokenCount']),
            'packedSequences': int(counters['packedSequences']),
            'sampleBatches': int(counters['sampleBatches']),
            'selectedLayers': ','.join(str(x) for x in selected_layers),
            'topK': int(layers[0].mlp.top_k if len(layers) > 0 else 0),
            'numExpertsByLayer': {str(k): int(v) for k, v in num_experts_by_layer.items()},
            'renormTopK': bool(args.renorm_topk),
            'layerWise': bool(args.layer_wise or collect_mode == 'replay_per_layer'),
            'collectMode': str(collect_mode),
            'lazyLoad': bool(args.lazy_load),
            'batchSize': int(args.batch_size) if args.batch_size is not None else 0,
            'sampleBatchSize': int(args.sample_batch_size),
            'packSamples': bool(args.pack_samples),
            'dataset': str(args.dataset) if args.dataset is not None else '',
            'datasetFile': str(args.dataset_file) if args.dataset_file is not None else '',
            'datasetFormat': str(args.dataset_format) if args.dataset_format is not None else '',
            'datasetSplit': str(args.dataset_split) if args.dataset_split is not None else '',
            'datasetTextField': str(args.dataset_text_field) if args.dataset_text_field is not None else '',
            'datasetMessagesField': str(args.dataset_messages_field) if args.dataset_messages_field is not None else '',
            'minSamples': int(args.min_samples),
            'maxSamples': int(args.max_samples),
        }
    }

    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
        f.write(NEWLINE)

    print(args.output)

if __name__ == '__main__':
    main()
`;
}

function parseLayersSpec(spec: string | undefined, maxLayers?: number): number[] {
  if (!spec || spec.trim().length === 0) {
    if (typeof maxLayers !== 'number') {
      return [];
    }
    return Array.from({ length: maxLayers }, (_, index) => index);
  }

  const result = new Set<number>();
  for (const rawPart of spec.split(',')) {
    const part = rawPart.trim();
    if (part.length === 0) {
      continue;
    }

    if (part.includes('-')) {
      const [startToken, endToken] = part.split('-', 2);
      const start = assertInteger(startToken, 'layers.start', 0, 100_000);
      const end = assertInteger(endToken, 'layers.end', 0, 100_000);
      const step = start <= end ? 1 : -1;
      for (let value = start; step > 0 ? value <= end : value >= end; value += step) {
        if (typeof maxLayers === 'number' && value >= maxLayers) {
          continue;
        }
        result.add(value);
      }
      continue;
    }

    const value = assertInteger(part, 'layers.value', 0, 100_000);
    if (typeof maxLayers === 'number' && value >= maxLayers) {
      continue;
    }
    result.add(value);
  }

  return [...result].sort((left, right) => left - right);
}

function normalizeCollectMode(config: MlxCollectConfig): CollectMode {
  const value = config.collectMode ?? (config.layerWise ? 'replay_per_layer' : 'single_pass');
  const allowed = new Set<CollectMode>([
    'single_pass',
    'replay_per_layer',
    'reload_per_layer'
  ]);

  if (!allowed.has(value)) {
    throw new Error(
      `Invalid collectMode: ${value}. Expected single_pass, replay_per_layer, or reload_per_layer`
    );
  }

  return value;
}

async function inferLayerCountFromConfig(modelPath: string): Promise<number | undefined> {
  try {
    const configPath = path.resolve(modelPath, 'config.json');
    const raw = await readJsonFileSafe<Record<string, unknown>>(configPath);
    const candidate =
      raw.num_hidden_layers ??
      raw.n_layers ??
      raw.n_layer ??
      raw.num_layers ??
      raw.layers;

    if (typeof candidate === 'number' && Number.isInteger(candidate) && candidate > 0) {
      return candidate;
    }

    if (typeof candidate === 'string' && candidate.trim().length > 0) {
      const parsed = Number(candidate);
      if (Number.isInteger(parsed) && parsed > 0) {
        return parsed;
      }
    }
  } catch {
    // Best effort only.
  }

  return undefined;
}

function assertSingleInputSource(config: MlxCollectConfig): void {
  const sources = [
    typeof config.prompt === 'string' && config.prompt.trim().length > 0,
    typeof config.datasetName === 'string' && config.datasetName.trim().length > 0,
    typeof config.datasetFile === 'string' && config.datasetFile.trim().length > 0
  ].filter(Boolean).length;

  if (sources !== 1) {
    throw new Error('collect requires exactly one of prompt, datasetName, or datasetFile');
  }
}

function normalizeDatasetFormat(value: DatasetFormat | undefined): DatasetFormat {
  const normalized = value ?? 'auto';
  const allowed = new Set<DatasetFormat>(['auto', 'json', 'jsonl', 'csv', 'parquet', 'text']);

  if (!allowed.has(normalized)) {
    throw new Error(
      `Invalid datasetFormat: ${normalized}. Expected auto, json, jsonl, csv, parquet, or text`
    );
  }

  return normalized;
}

export function buildMlxCollectArgs(config: MlxCollectConfig, telemetryPath: string): string[] {
  assertSingleInputSource(config);

  const maxTokens = assertInteger(config.maxTokens ?? 256, 'maxTokens', 1, 8192);
  const maxSamples = assertInteger(config.maxSamples ?? 100, 'maxSamples', 1, 100_000);
  const minSamples = assertInteger(config.minSamples ?? 1, 'minSamples', 1, 100_000);
  const batchSize =
    typeof config.batchSize === 'number'
      ? assertInteger(config.batchSize, 'batchSize', 1, 8192)
      : undefined;
  const sampleBatchSize = assertInteger(
    config.sampleBatchSize ?? 1,
    'sampleBatchSize',
    1,
    1024
  );

  if (minSamples > maxSamples) {
    throw new Error('minSamples cannot exceed maxSamples');
  }

  const prompt =
    typeof config.prompt === 'string' && config.prompt.trim().length > 0 ? config.prompt : undefined;
  const datasetName =
    typeof config.datasetName === 'string' && config.datasetName.trim().length > 0
      ? config.datasetName
      : undefined;
  const datasetFile =
    typeof config.datasetFile === 'string' && config.datasetFile.trim().length > 0
      ? path.resolve(config.datasetFile)
      : undefined;

  const datasetFormat = normalizeDatasetFormat(config.datasetFormat);
  const resolvedCollectMode = normalizeCollectMode(config);
  const processCollectMode =
    resolvedCollectMode === 'reload_per_layer' ? 'replay_per_layer' : resolvedCollectMode;

  return [
    '-c',
    pythonScript(),
    '--model',
    path.resolve(config.modelPath),
    ...(prompt ? ['--prompt', prompt] : []),
    ...(datasetName ? ['--dataset', datasetName] : []),
    ...(datasetFile ? ['--dataset-file', datasetFile] : []),
    '--dataset-format',
    datasetFormat,
    '--dataset-split',
    config.datasetSplit ?? 'train',
    '--dataset-text-field',
    config.datasetTextField ?? '',
    '--dataset-messages-field',
    config.datasetMessagesField ?? '',
    '--max-samples',
    String(maxSamples),
    '--min-samples',
    String(minSamples),
    '--max-tokens',
    String(maxTokens),
    '--sample-batch-size',
    String(sampleBatchSize),
    '--layers',
    config.includeLayers ?? '',
    ...(config.renormTopK ? ['--renorm-topk'] : []),
    ...(config.layerWise || resolvedCollectMode === 'replay_per_layer' ? ['--layer-wise'] : []),
    '--collect-mode',
    processCollectMode,
    ...(typeof batchSize === 'number' ? ['--batch-size', String(batchSize)] : []),
    ...(config.packSamples ? ['--pack-samples'] : []),
    ...(config.lazyLoad ? ['--lazy-load'] : []),
    '--output',
    telemetryPath
  ];
}

export const __testOnly = {
  buildMlxCollectArgs,
  pythonScript
};

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

async function runCollectorProcess(
  config: MlxCollectConfig,
  telemetryPath: string
): Promise<{ telemetryPath: string; telemetry: ModelTelemetry }> {
  const pythonBin = config.pythonBin && config.pythonBin.trim().length > 0 ? config.pythonBin : 'python3';

  const args = buildMlxCollectArgs(config, telemetryPath);
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

function mergeReloadedLayerTelemetries(
  parts: Array<{ telemetryPath: string; telemetry: ModelTelemetry }>,
  config: MlxCollectConfig,
  selectedLayers: number[]
): ModelTelemetry {
  const experts = parts
    .flatMap((part) => part.telemetry.experts)
    .sort((left, right) => {
      if (left.layer !== right.layer) {
        return left.layer - right.layer;
      }
      return left.expert - right.expert;
    });

  const base = parts[0]?.telemetry;
  if (!base) {
    throw new Error('reload_per_layer produced no telemetry parts');
  }

  return {
    modelName: base.modelName,
    experts,
    metadata: {
      ...(base.metadata ?? {}),
      selectedLayers: selectedLayers.join(','),
      layerWise: true,
      collectMode: 'reload_per_layer',
      lazyLoad: config.lazyLoad ?? true,
      mergedTelemetryFiles: parts.length
    }
  };
}

export async function collectTelemetryWithMlx(
  config: MlxCollectConfig
): Promise<{ telemetryPath: string; telemetry: ModelTelemetry }> {
  const outputDir = path.resolve(config.outputDir);
  await ensureSecureDir(outputDir);
  assertSingleInputSource(config);

  const resolvedCollectMode = normalizeCollectMode(config);

  if (resolvedCollectMode === 'reload_per_layer') {
    const inferredLayerCount = await inferLayerCountFromConfig(config.modelPath);
    if (typeof inferredLayerCount !== 'number' && !config.includeLayers) {
      throw new Error(
        'reload_per_layer requires --layers unless layer count can be inferred from config.json'
      );
    }

    const selectedLayers = parseLayersSpec(config.includeLayers, inferredLayerCount);
    if (selectedLayers.length === 0) {
      throw new Error('reload_per_layer resolved zero selected layers');
    }

    const partials: Array<{ telemetryPath: string; telemetry: ModelTelemetry }> = [];
    for (const layer of selectedLayers) {
      const partialTelemetryPath = resolveSafePath(
        outputDir,
        `telemetry-layer-${layer}-${randomUUID()}.json`
      );
      const partial = await runCollectorProcess(
        {
          ...config,
          includeLayers: String(layer),
          collectMode: 'replay_per_layer',
          layerWise: true,
          ...(config.lazyLoad === false ? {} : { lazyLoad: true })
        },
        partialTelemetryPath
      );
      partials.push(partial);
    }

    const telemetryPath = resolveSafePath(outputDir, `telemetry-${randomUUID()}.json`);
    const telemetry = mergeReloadedLayerTelemetries(partials, config, selectedLayers);
    await writeJsonAtomicSafe(telemetryPath, telemetry);
    return {
      telemetryPath,
      telemetry
    };
  }

  const telemetryPath = resolveSafePath(outputDir, `telemetry-${randomUUID()}.json`);
  return runCollectorProcess(config, telemetryPath);
}
