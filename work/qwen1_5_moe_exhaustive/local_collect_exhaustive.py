#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def parse_layers(spec: str | None, max_layers: int) -> list[int]:
    if spec is None or spec.strip() == "":
        return list(range(max_layers))

    result: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            left, right = part.split('-', 1)
            start = int(left)
            end = int(right)
            step = 1 if start <= end else -1
            result.extend(list(range(start, end + step, step)))
        else:
            result.append(int(part))

    return [layer for layer in sorted(set(result)) if 0 <= layer < max_layers]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise RuntimeError(f'Line {line_number} is not a JSON object')
            yield record


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def update_and_trace(
    *,
    sample_index: int,
    sample_id: str,
    token_ids: list[int],
    layer_id: int,
    layer,
    hidden,
    stats: dict[tuple[int, int], dict[str, float | int]],
    raw_writer,
    renorm_topk: bool,
    batch_size: int | None,
) -> None:
    h = layer.input_layernorm(hidden)
    flat_h = h.reshape(-1, h.shape[-1])
    total_tokens = int(flat_h.shape[0])

    gate = layer.mlp.gate
    switch = layer.mlp.switch_mlp
    num_experts = int(layer.mlp.num_experts)
    top_k = int(layer.mlp.top_k)

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
        inds = mx.argpartition(-routing_weights, kth=top_k - 1, axis=-1)[..., :top_k]
        mx.eval(inds, routing_weights)

        topk_weights = mx.take_along_axis(routing_weights, inds, axis=-1)
        if renorm_topk:
            denom = mx.maximum(mx.sum(topk_weights, axis=-1, keepdims=True), 1e-12)
            routing_weights = routing_weights / denom
            topk_weights = mx.take_along_axis(routing_weights, inds, axis=-1)
        mx.eval(topk_weights, routing_weights)

        topk_experts = inds.tolist()
        topk_weight_rows = topk_weights.tolist()

        token_route_meta: list[dict[int, tuple[int, float]]] = [dict() for _ in range(num_tokens)]
        token_sets: list[list[int]] = [[] for _ in range(num_experts)]
        for token_idx, (experts_row, weight_row) in enumerate(zip(topk_experts, topk_weight_rows, strict=True)):
            ranked_pairs = sorted(
                [(int(expert_id), float(weight)) for expert_id, weight in zip(experts_row, weight_row, strict=True)],
                key=lambda item: item[1],
                reverse=True,
            )
            for rank, (expert_id, selected_weight) in enumerate(ranked_pairs, start=1):
                token_route_meta[token_idx][expert_id] = (rank, selected_weight)
                token_sets[expert_id].append(token_idx)

        for expert_id in range(num_experts):
            active_tokens = token_sets[expert_id]
            if not active_tokens:
                continue

            idx = mx.array(active_tokens)
            active_h = chunk_h[idx]

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

            ean_list = [float(value) for value in ean_norm.tolist()]
            rw_list = [float(value) for value in active_rw.tolist()]
            weighted_list = [float(value) for value in weighted_norm.tolist()]

            key = (layer_id, expert_id)
            entry = stats.get(key)
            if entry is None:
                entry = {
                    'activeTokenCount': 0,
                    'gateValueSum': 0.0,
                    'activationNormSum': 0.0,
                    'weightedActivationNormSum': 0.0,
                    'maxActivation': 0.0,
                }
                stats[key] = entry

            for rel_idx, token_idx in enumerate(active_tokens):
                abs_token_idx = start + token_idx
                rank, selected_weight = token_route_meta[token_idx][expert_id]
                event = {
                    'sampleIndex': sample_index,
                    'sampleId': sample_id,
                    'layer': layer_id,
                    'expert': expert_id,
                    'tokenIndex': abs_token_idx,
                    'tokenId': int(token_ids[abs_token_idx]),
                    'topKRank': rank,
                    'routerWeight': round(rw_list[rel_idx], 8),
                    'selectedWeight': round(selected_weight, 8),
                    'activationNorm': round(ean_list[rel_idx], 8),
                    'weightedActivationNorm': round(weighted_list[rel_idx], 8),
                }
                raw_writer.write(json.dumps(event, ensure_ascii=False) + '\n')

            entry['activeTokenCount'] = int(entry['activeTokenCount']) + len(active_tokens)
            entry['gateValueSum'] = float(entry['gateValueSum']) + sum(rw_list)
            entry['activationNormSum'] = float(entry['activationNormSum']) + sum(ean_list)
            entry['weightedActivationNormSum'] = float(entry['weightedActivationNormSum']) + sum(weighted_list)
            entry['maxActivation'] = max(float(entry['maxActivation']), max(ean_list))


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect exhaustive MLX activation trace + aggregate telemetry from a JSONL dataset.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset-jsonl', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--text-field', default='text')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--layers', default='')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--renorm-topk', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_trace_path = output_dir / 'activation-trace.jsonl'
    telemetry_path = output_dir / 'telemetry.json'
    samples_path = output_dir / 'sample-manifest.jsonl'
    summary_path = output_dir / 'collector-summary.json'

    dataset_path = Path(args.dataset_jsonl).resolve()
    model, tokenizer = load(args.model, lazy=False)
    layers = model.model.layers
    selected_layers = parse_layers(args.layers, len(layers))
    selected_layer_set = set(selected_layers)

    stats: dict[tuple[int, int], dict[str, float | int]] = {}
    processed_samples = 0
    total_prompt_tokens = 0
    num_experts_by_layer: dict[str, int] = {}

    ensure_parent(raw_trace_path)
    ensure_parent(telemetry_path)
    ensure_parent(samples_path)
    ensure_parent(summary_path)

    with raw_trace_path.open('w', encoding='utf-8') as raw_writer, samples_path.open('w', encoding='utf-8') as sample_writer:
        for row in iter_jsonl(dataset_path):
            if processed_samples >= args.max_samples:
                break
            text = str(row.get(args.text_field, '')).strip()
            if not text:
                continue
            sample_id = str(row.get('id', processed_samples))
            token_ids = tokenizer.encode(text)
            if len(token_ids) > args.max_tokens:
                token_ids = token_ids[:args.max_tokens]
            if not token_ids:
                continue

            total_prompt_tokens += len(token_ids)
            sample_writer.write(json.dumps({
                'sampleIndex': processed_samples,
                'sampleId': sample_id,
                'textField': args.text_field,
                'tokenCount': len(token_ids),
            }) + '\n')

            x = mx.array([token_ids])
            hidden = model.model.embed_tokens(x)

            mask = None
            if x.shape[1] > 1:
                mask = mx.full((x.shape[1], x.shape[1]), -1e9)
                mask = mx.triu(mask, k=1)
                mask = mask.astype(hidden.dtype)

            local_hidden = hidden
            for layer_id, layer in enumerate(layers):
                if layer_id in selected_layer_set:
                    update_and_trace(
                        sample_index=processed_samples,
                        sample_id=sample_id,
                        token_ids=token_ids,
                        layer_id=layer_id,
                        layer=layer,
                        hidden=local_hidden,
                        stats=stats,
                        raw_writer=raw_writer,
                        renorm_topk=args.renorm_topk,
                        batch_size=args.batch_size,
                    )
                    num_experts_by_layer[str(layer_id)] = int(layer.mlp.num_experts)
                local_hidden = layer(local_hidden, mask=mask)

            processed_samples += 1

    experts = []
    for layer_id in selected_layers:
        layer_num_experts = int(layers[layer_id].mlp.num_experts)
        num_experts_by_layer[str(layer_id)] = layer_num_experts
        for expert_id in range(layer_num_experts):
            entry = stats.get((layer_id, expert_id), {
                'activeTokenCount': 0,
                'gateValueSum': 0.0,
                'activationNormSum': 0.0,
                'weightedActivationNormSum': 0.0,
                'maxActivation': 0.0,
            })
            active = int(entry['activeTokenCount'])
            denom = max(active, 1)
            gate_sum = float(entry['gateValueSum'])
            act_sum = float(entry['activationNormSum'])
            weighted_sum = float(entry['weightedActivationNormSum'])
            experts.append({
                'layer': layer_id,
                'expert': expert_id,
                'activeTokenCount': active,
                'tokenCount': active,
                'gateValueSum': round(gate_sum, 8),
                'activationNormSum': round(act_sum, 8),
                'weightedActivationNormSum': round(weighted_sum, 8),
                'averageGateValue': round(gate_sum / denom, 8),
                'averageActivationNorm': round(act_sum / denom, 8),
                'activationScore': round(weighted_sum / denom, 8),
                'frequency': active,
                'eanSum': round(act_sum, 8),
                'eanMean': round(act_sum / denom, 8),
                'weightedEanSum': round(weighted_sum, 8),
                'reap': round(weighted_sum / denom, 8),
                'maxActivation': round(float(entry['maxActivation']), 8),
            })

    telemetry = {
        'modelName': Path(str(args.model)).name,
        'experts': experts,
        'metadata': {
            'source': 'mlx_exhaustive_jsonl_collector',
            'datasetJsonl': str(dataset_path),
            'textField': args.text_field,
            'processedSamples': processed_samples,
            'promptLengthTokens': total_prompt_tokens,
            'selectedLayers': ','.join(str(x) for x in selected_layers),
            'topK': int(layers[0].mlp.top_k if len(layers) > 0 else 0),
            'numExpertsByLayer': num_experts_by_layer,
            'renormTopK': bool(args.renorm_topk),
            'batchSize': int(args.batch_size) if args.batch_size is not None else 0,
            'activationTracePath': str(raw_trace_path),
            'sampleManifestPath': str(samples_path),
        },
    }
    summary = {
        'model': args.model,
        'datasetJsonl': str(dataset_path),
        'processedSamples': processed_samples,
        'totalPromptTokens': total_prompt_tokens,
        'selectedLayers': selected_layers,
        'rawTracePath': str(raw_trace_path),
        'telemetryPath': str(telemetry_path),
    }

    with telemetry_path.open('w', encoding='utf-8') as handle:
        json.dump(telemetry, handle, indent=2)
        handle.write('\n')
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
