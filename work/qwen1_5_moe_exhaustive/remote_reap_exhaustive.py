#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from reap.observer import MoETransformerObserver, MoETransformerObserverConfig
from reap.model_util import MODEL_ATTRS


@dataclass
class Qwen2MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: str | None = 'Qwen2MoeSparseMoeBlock'
    num_experts_attr_name: str = 'num_experts'
    top_k_attr_name: str = 'top_k'
    fused_experts: bool = False


class TraceContext:
    sample_index: int = -1
    sample_id: str = ''
    token_ids: list[int] = []


def iter_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise RuntimeError(f'Line {line_number} is not an object')
            yield record


def detect_qwen_moe_mode(model) -> dict[str, Any]:
    for module in model.modules():
        if module.__class__.__name__ != 'Qwen2MoeSparseMoeBlock':
            continue
        experts = getattr(module, 'experts', None)
        is_fused = not isinstance(experts, torch.nn.ModuleList)
        return {
            'fused_experts': is_fused,
            'num_experts_attr_name': 'num_experts',
            'top_k_attr_name': 'num_experts_per_tok' if is_fused else 'top_k',
        }
    raise RuntimeError('No Qwen2MoeSparseMoeBlock modules found')


def adapt_qwen2_moe_blocks(model, mode: dict[str, Any]):
    for module in model.modules():
        if module.__class__.__name__ != 'Qwen2MoeSparseMoeBlock':
            continue
        if mode['fused_experts'] and not hasattr(module, 'router'):
            module.router = module.gate
        if mode['fused_experts']:
            module.num_experts = int(module.experts.num_experts)
            module.num_experts_per_tok = int(module.gate.top_k)
        else:
            module.num_experts = int(module.num_experts)
            module.top_k = int(module.top_k)


def register_raw_hooks(model, raw_writer, ctx: TraceContext, *, fused_experts: bool):
    hooks = []

    def make_hook(layer_number: int):
        @torch.no_grad()
        def _hook(module, args, output):
            input_tensor = args[0]
            device = input_tensor.device
            batch_size, sequence_length, hidden_dim = input_tensor.shape
            flat_input = input_tensor.view(-1, hidden_dim)

            num_experts = int(module.num_experts)
            top_k = int(module.num_experts_per_tok) if fused_experts else int(module.top_k)
            if fused_experts:
                _, router_scores = output
                router_logits = module.router(flat_input)
                _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
                selected_experts = selected_experts.to(device)

                router_indices = (
                    torch.arange(batch_size * sequence_length, device=device)
                    .view(1, -1)
                    .expand(router_scores.size(0), -1)
                )
                router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
                routed_in = torch.gather(flat_input, dim=0, index=router_indices).to(device)
                routed_out = module.experts(routed_in)
                activations = routed_out.view(num_experts, *flat_input.shape)
            else:
                *_, router_logits = output
                _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
                selected_experts = selected_experts.to(device)
                activations = torch.zeros((num_experts, *flat_input.shape), device=device)
                for expert_id, expert in enumerate(module.experts):
                    activations[expert_id] = expert(flat_input).to(device)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float).to(device)
            topk_weights = torch.gather(routing_weights, 1, selected_experts)

            token_route_meta: list[dict[int, tuple[int, float]]] = [dict() for _ in range(selected_experts.shape[0])]
            for token_idx in range(selected_experts.shape[0]):
                ranked_pairs = sorted(
                    [
                        (int(selected_experts[token_idx, rank].item()), float(topk_weights[token_idx, rank].item()))
                        for rank in range(selected_experts.shape[1])
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )
                for rank, (expert_id, selected_weight) in enumerate(ranked_pairs, start=1):
                    token_route_meta[token_idx][expert_id] = (rank, selected_weight)

            for expert_id in range(num_experts):
                active_mask = (selected_experts == expert_id).any(dim=-1)
                if not active_mask.any():
                    continue
                active_router_weights = routing_weights[active_mask, expert_id]
                ean_norm = torch.linalg.norm(activations[expert_id, active_mask, :], dim=-1)
                weighted = ean_norm * active_router_weights
                active_indices = active_mask.nonzero(as_tuple=False).view(-1).tolist()
                rw_list = active_router_weights.detach().cpu().tolist()
                ean_list = ean_norm.detach().cpu().tolist()
                weighted_list = weighted.detach().cpu().tolist()

                for rel_idx, token_idx in enumerate(active_indices):
                    abs_token_idx = int(token_idx)
                    token_id = int(ctx.token_ids[abs_token_idx]) if abs_token_idx < len(ctx.token_ids) else -1
                    rank, selected_weight = token_route_meta[token_idx][expert_id]
                    raw_writer.write(json.dumps({
                        'sampleIndex': ctx.sample_index,
                        'sampleId': ctx.sample_id,
                        'layer': layer_number,
                        'expert': expert_id,
                        'tokenIndex': abs_token_idx,
                        'tokenId': token_id,
                        'topKRank': rank,
                        'routerWeight': round(float(rw_list[rel_idx]), 8),
                        'selectedWeight': round(float(selected_weight), 8),
                        'activationNorm': round(float(ean_list[rel_idx]), 8),
                        'weightedActivationNorm': round(float(weighted_list[rel_idx]), 8),
                    }) + '\n')
        return _hook

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Qwen2MoeSparseMoeBlock':
            layer_number = int(name.split('.')[2]) if '.layers.' in name else int(name.split('.')[-1])
            hooks.append(module.register_forward_hook(make_hook(layer_number)))
    return hooks


def tensor_to_list(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def apply_qwen2_prune(
    *,
    model,
    observer_data,
    prune_method: str,
    n_experts_to_prune: int,
    pruned_model_dir: Path,
):
    for layer, layer_state in observer_data.items():
        if prune_method == 'frequency':
            saliency = layer_state['expert_frequency'].detach().cpu().float()
        else:
            saliency = layer_state[prune_method].detach().cpu().float()
        _, experts_to_prune = torch.topk(saliency, n_experts_to_prune, largest=False)
        prune_set = set(experts_to_prune.tolist())

        moe = model.model.layers[layer].mlp
        total_experts = int(moe.experts.num_experts) if hasattr(moe.experts, 'num_experts') else int(moe.num_experts)
        retained = [index for index in range(total_experts) if index not in prune_set]
        if not retained:
            raise RuntimeError(f'Layer {layer} would retain zero experts')

        if isinstance(moe.experts, torch.nn.ModuleList):
            moe.experts = torch.nn.ModuleList([moe.experts[index] for index in retained])
        else:
            moe.experts.gate_up_proj.data = moe.experts.gate_up_proj[retained]
            moe.experts.down_proj.data = moe.experts.down_proj[retained]
            moe.experts.num_experts = len(retained)
        moe.num_experts = len(retained)
        moe.gate.weight.data = moe.gate.weight.data[retained]
        moe.gate.out_features = len(retained)

    model.config.num_experts = len(retained)
    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(pruned_model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run exhaustive upstream REAP trace + pruning on Qwen1.5-MoE-A2.7B-Chat with a JSONL dataset.')
    parser.add_argument('--model', default='Qwen/Qwen1.5-MoE-A2.7B-Chat')
    parser.add_argument('--dataset-jsonl', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--text-field', default='text')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--compression-ratio', type=float, default=0.25)
    parser.add_argument('--prune-method', default='reap')
    args = parser.parse_args()

    MODEL_ATTRS['Qwen2MoeForCausalLM'] = {
        'moe_block': 'mlp',
        'gate_proj': 'gate_up_proj',
        'up_proj': 'gate_up_proj',
        'down_proj': 'down_proj',
        'experts': 'experts',
        'fused': True,
        'router': 'gate',
        'num_experts': 'num_experts',
        'num_experts_per_tok': 'num_experts_per_tok',
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_trace_path = output_dir / 'activation-trace.jsonl'
    sample_manifest_path = output_dir / 'sample-manifest.jsonl'
    observer_state_path = output_dir / 'observer-state.pt'
    observer_summary_path = output_dir / 'observer-summary.json'
    decision_trace_path = output_dir / 'decision-trace.jsonl'
    decision_summary_path = output_dir / 'decision-summary.json'

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_dtype = torch.float16 if load_device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=load_dtype,
        trust_remote_code=True,
    )
    model.to(load_device)
    model.eval()
    observer_mode = detect_qwen_moe_mode(model)
    adapt_qwen2_moe_blocks(model, observer_mode)
    primary_device = torch.device(load_device)

    ctx = TraceContext()
    observer = MoETransformerObserver(
        model,
        hook_config=Qwen2MoEObserverHookConfig(
            num_experts_attr_name=observer_mode['num_experts_attr_name'],
            top_k_attr_name=observer_mode['top_k_attr_name'],
            fused_experts=observer_mode['fused_experts'],
        ),
    )

    processed_samples = 0
    total_tokens = 0
    with activation_trace_path.open('w', encoding='utf-8') as raw_writer, sample_manifest_path.open('w', encoding='utf-8') as sample_writer:
        raw_hooks = register_raw_hooks(
            model,
            raw_writer,
            ctx,
            fused_experts=observer_mode['fused_experts'],
        )
        try:
            for row in iter_jsonl(Path(args.dataset_jsonl)):
                if processed_samples >= args.max_samples:
                    break
                text = str(row.get(args.text_field, '')).strip()
                if not text:
                    continue
                encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=args.max_tokens)
                input_ids = encoded['input_ids']
                if input_ids.numel() == 0:
                    continue
                token_ids = input_ids[0].tolist()
                ctx.sample_index = processed_samples
                ctx.sample_id = str(row.get('id', processed_samples))
                ctx.token_ids = token_ids
                sample_writer.write(json.dumps({
                    'sampleIndex': processed_samples,
                    'sampleId': ctx.sample_id,
                    'tokenCount': len(token_ids),
                }) + '\n')
                total_tokens += len(token_ids)
                model(input_ids.to(primary_device))
                processed_samples += 1
        finally:
            for hook in raw_hooks:
                hook.remove()

    observer_data = observer.report_state()
    torch.save(observer_data, observer_state_path)

    summary_layers = {}
    for layer, layer_state in observer_data.items():
        summary_layers[str(layer)] = {
            key: tensor_to_list(value)
            for key, value in layer_state.items()
            if key in {'total_tokens', 'expert_frequency', 'ean_sum', 'weighted_ean_sum', 'reap', 'max_activations'}
        }
    with observer_summary_path.open('w', encoding='utf-8') as handle:
        json.dump({
            'model': args.model,
            'datasetJsonl': str(Path(args.dataset_jsonl).resolve()),
            'processedSamples': processed_samples,
            'totalTokens': total_tokens,
            'layers': summary_layers,
        }, handle, indent=2)
        handle.write('\n')

    first_layer = observer_data[next(iter(observer_data))]
    total_experts = len(first_layer['expert_frequency'])
    n_experts_to_prune = int(total_experts * args.compression_ratio)

    with decision_trace_path.open('w', encoding='utf-8') as trace_handle:
        decision_summary = {'pruneMethod': args.prune_method, 'compressionRatio': args.compression_ratio, 'perLayer': {}}
        for layer, layer_state in observer_data.items():
            if args.prune_method == 'frequency':
                saliency = layer_state['expert_frequency'].detach().cpu().float()
            else:
                saliency = layer_state[args.prune_method].detach().cpu().float()
            values, indices = torch.sort(saliency, descending=False)
            prune_set = set(indices[:n_experts_to_prune].tolist())
            decision_summary['perLayer'][str(layer)] = {
                'numExperts': total_experts,
                'nExpertsToPrune': n_experts_to_prune,
                'threshold': float(values[n_experts_to_prune - 1].item()) if n_experts_to_prune > 0 else None,
            }
            for rank, expert_idx in enumerate(indices.tolist(), start=1):
                trace_handle.write(json.dumps({
                    'layer': int(layer),
                    'expert': int(expert_idx),
                    'rank': rank,
                    'signal': round(float(saliency[expert_idx].item()), 8),
                    'pruned': expert_idx in prune_set,
                    'reason': 'low_signal_pruned' if expert_idx in prune_set else 'high_signal_retained',
                }) + '\n')
    with decision_summary_path.open('w', encoding='utf-8') as handle:
        json.dump(decision_summary, handle, indent=2)
        handle.write('\n')

    pruned_model_dir = output_dir / 'pruned-model'
    apply_qwen2_prune(
        model=model,
        observer_data=observer_data,
        prune_method=args.prune_method,
        n_experts_to_prune=n_experts_to_prune,
        pruned_model_dir=pruned_model_dir,
    )

    print(json.dumps({
        'outputDir': str(output_dir),
        'activationTracePath': str(activation_trace_path),
        'observerStatePath': str(observer_state_path),
        'decisionTracePath': str(decision_trace_path),
        'prunedModelDir': str(pruned_model_dir),
        'processedSamples': processed_samples,
        'totalTokens': total_tokens,
        'nExpertsToPrune': n_experts_to_prune,
    }, indent=2))


if __name__ == '__main__':
    main()
