# reap-mlx

Port of [Cerebras REAP](https://arxiv.org/abs/2510.13999) (Router-weighted Expert Activation Pruning) to Apple Silicon. Prune experts from Mixture-of-Experts LLMs locally on your Mac using [MLX](https://github.com/ml-explore/mlx) -- no CUDA required.

The original REAP paper and [reference implementation](https://github.com/CerebrasResearch/reap) target multi-GPU setups (A100/H100). This tool does the same thing on a MacBook: collect routing telemetry from an MLX model, score experts by the REAP saliency criterion, and structurally remove the lowest-scoring ones from the checkpoint.

## Pipeline

```
collect  →  run  →  apply
```

**collect** runs a forward pass through an MLX MoE model and records per-expert gate values and activation norms. **run** reads that telemetry, scores experts, and writes a pruning plan. **apply** slices the pruned experts out of the checkpoint and saves a smaller model.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Node 20+, pnpm
- Python 3.11+ with `mlx` and `mlx_lm` installed

## Quick start

```bash
pnpm install && pnpm build
```

Collect routing telemetry from a local MoE model:

```bash
node dist/cli/index.js collect \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --output ./tmp \
  --prompt "Explain sparse MoE routing." \
  --layers 0-3
```

Build a pruning plan (prune 5% of experts per layer):

```bash
node dist/cli/index.js run \
  --model ./tmp/telemetry-*.json \
  --output ./tmp/plan \
  --ratio 0.05 \
  --no-legacy
```

Apply it to the checkpoint:

```bash
node dist/cli/index.js apply \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --plan ./tmp/plan/pruning-plan.json \
  --output ./tmp/pruned-model
```

Use `--dry-run` on `apply` to validate the plan without writing files.

## How saliency scoring works

Each expert gets a score:

```
saliency_j = mean(g_j(x) * ||f_j(x)||)
```

`g_j(x)` is the softmax gating weight for expert j on input x. `||f_j(x)||` is the L2 norm of expert j's output. The mean is taken over tokens that were actually routed to that expert. This matches the Cerebras REAP paper's formulation -- experts that receive low routing weight and produce small activations are pruned first.

The `collect` command captures `weightedActivationNormSum` (the numerator) and `activeTokenCount` (the denominator) per expert. When those fields are missing, the scorer falls back to `averageGateValue * averageActivationNorm`, or to raw `activationScore` if `--no-legacy` is not set.

Pruning is per-layer, not global. Each layer has at least `--min-experts` (default 1) experts kept alive.

## Commands

### collect

```
--model <dir>         Local MLX model directory
--output <dir>        Where to write telemetry JSON
--prompt <text>       Input text for the forward pass
--max-tokens <n>      Token cap (default: 256)
--layers <spec>       Layer filter, e.g. "0-3,8,10"
--renorm-topk         Normalize top-k gate weights to sum to 1
--python <bin>        Python binary (default: python3)
```

### run

```
--model <file>        Telemetry JSON from collect
--output <dir>        Output directory for plan + observation log
--ratio <0..0.95>     Target prune ratio per layer
--calibration <1..25> Calibration rounds (default: 2)
--min-experts <n>     Min experts kept per layer (default: 1)
--no-legacy           Require REAP saliency fields, reject activationScore fallback
--json                Print plan to stdout
```

### apply

```
--model <dir>         Source MLX model directory
--plan <file>         Pruning plan JSON from run
--output <dir>        Output pruned model directory
--dry-run             Validate without writing
```

### observe

Parses the JSONL observation log from a `run` and prints timing/stage summary.

```
--file <path>         Observation log file
--json                Machine-readable output
```

### init

Generates synthetic telemetry for testing without a real model.

```
--output <file>       Output telemetry JSON
--model-name <name>   Model name (default: synthetic-moe)
--layers <1..512>     Layer count (default: 8)
--experts <2..512>    Experts per layer (default: 8)
--seed <int>          RNG seed
```

## Supported architectures

Currently supports models using the `switch_mlp` MoE pattern (Qwen MoE, Mixtral-style). Other architectures (DeepSeek V2/V3, Llama 4, GLM MoE) are not yet supported.

## Relation to Cerebras REAP

The Cerebras repo includes expert merging (HC-SMoE, M-SMoE, SubMoE), a full evaluation harness (EvalPlus, LiveCodeBench, WildBench, lm-eval), and support for many architectures. This repo only does pruning, only on MLX, and does not include evaluation. Use the Cerebras repo if you have CUDA GPUs and want the full research pipeline.

## Development

```bash
pnpm verify    # typecheck + build + test
pnpm test      # build + vitest
pnpm lint      # typecheck only
```

## References

- Paper: [REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression](https://arxiv.org/abs/2510.13999)
- Reference implementation: [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
- MLX: [ml-explore/mlx](https://github.com/ml-explore/mlx)
