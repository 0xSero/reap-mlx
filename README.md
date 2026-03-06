# reap-mlx

REAP pruning for MLX MoE models on Apple Silicon.

If you want the shortest mental model:

```text
collect telemetry -> run planner -> apply pruning plan
```

This repo ports the pruning part of [Cerebras REAP](https://github.com/CerebrasResearch/reap) to local MLX workflows.

## 60-second quick start

Requirements:
- Apple Silicon Mac
- Node 20+, pnpm
- Python 3.11+
- `mlx` and `mlx_lm` installed

```bash
pnpm install
pnpm build
```

### 1) Collect telemetry

```bash
node dist/cli/index.js collect \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --output ./tmp \
  --prompt "Explain sparse MoE routing" \
  --max-tokens 512 \
  --layer-wise \
  --batch-size 128
```

### 2) Build pruning plan

```bash
node dist/cli/index.js run \
  --model ./tmp/telemetry-*.json \
  --output ./tmp/plan \
  --ratio 0.5 \
  --min-experts 1 \
  --no-legacy
```

### 3) Apply pruning

```bash
node dist/cli/index.js apply \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --plan ./tmp/plan/pruning-plan.json \
  --output ./tmp/pruned-model
```

Use `--dry-run` on `apply` to validate a plan without writing a new checkpoint.

## What each command does

- `collect`: runs one forward pass and saves per-expert routing + activation stats.
- `run`: scores experts from telemetry and writes `pruning-plan.json`.
- `parity`: runs the same prune config against two telemetry files and diffs the exact prune set.
- `apply`: physically removes pruned experts from the MLX checkpoint.
- `observe`: summarizes run observation logs.
- `init`: creates synthetic telemetry for local testing.

## REAP saliency used

For each expert `j`:

```text
saliency_j = mean( g_j(x) * ||f_j(x)|| )
```

- `g_j(x)`: router softmax weight for expert `j`
- `f_j(x)`: expert output
- Mean is over tokens routed to that expert

This is the REAP-style scoring used for pruning decisions in this project.

## CLI reference (minimal)

### collect

```text
--model <dir>       MLX model directory
--output <dir>      Telemetry output directory
--prompt <text>     Calibration text
--dataset <name>    HuggingFace dataset name
--dataset-split     Dataset split (default: train)
--dataset-text-field Field to read text from (default: instruction)
--max-samples <n>   Max dataset samples to aggregate (default: 100)
--max-tokens <n>    Token cap (default: 256)
--layers <spec>     Example: "0-3,8,10"
--renorm-topk       Renormalize top-k gate weights
--layer-wise        Re-run forward passes per selected layer before scoring
--batch-size <n>    Chunk flattened token activations during scoring
--python <bin>      Python binary (default: python3)
```

`--layer-wise` is useful when you want per-layer telemetry with minimal cross-layer coupling. `--batch-size` trades throughput for lower peak memory during expert scoring.

### full

```text
--model <dir>       MLX model directory
--output <dir>      Pipeline output directory
--dataset <name>    HuggingFace dataset name
--dataset-split     Dataset split (default: train)
--dataset-text-field Field to read text from (default: instruction)
--max-samples <n>   Max dataset samples to aggregate (default: 100)
--max-tokens <n>    Token cap (default: 256)
--layers <spec>     Example: "0-3,8,10"
--renorm-topk       Renormalize top-k gate weights
--layer-wise        Re-run forward passes per selected layer before scoring
--batch-size <n>    Chunk flattened token activations during scoring
--ratio <0..0.95>   Target prune ratio per layer
--min-experts <n>   Minimum experts kept per layer
--dry-run           Validate plan only
```

### run

```text
--model <file>      Telemetry JSON from collect
--output <dir>      Plan + observation output directory
--ratio <0..0.95>   Target prune ratio per layer
--calibration <n>   Calibration rounds (default: 2)
--min-experts <n>   Minimum experts kept per layer
--no-legacy         Disable fallback saliency fields
--json              Print plan JSON to stdout
```

### parity

```text
--left <file>                      Left telemetry JSON
--right <file>                     Right telemetry JSON
--output <dir>                     Output directory for left/right plans + parity report
--ratio <0..0.95>                  Target prune ratio per layer
--n-experts-to-prune-per-layer <n> Prune exactly n experts per layer
--prune-method <name>              reap|frequency|ean_sum|ean_mean|weighted_ean_sum
--require-identical-telemetry      Fail unless normalized telemetry hashes match exactly
--json                             Print parity report JSON to stdout
```

Use `parity` when the question is:

```text
same exact telemetry + same prune config => same exact experts pruned?
```

It writes `parity-report.json` and `parity-report.md` with:
- normalized telemetry hashes
- first differing expert row
- exact prune-set diff
- per-layer expert deltas

### apply

```text
--model <dir>       Source MLX model
--plan <file>       pruning-plan.json
--output <dir>      Pruned model output
--dry-run           Validate plan only
```

### observe

```text
--file <path>       Observation log file
--json              JSON output
```

### init

```text
--output <file>     Synthetic telemetry output
--model-name <name> Default: synthetic-moe
--layers <n>        Default: 8
--experts <n>       Default: 8
--seed <int>        RNG seed
```

## Current scope and limits

- Supports `switch_mlp`-style MLX MoE checkpoints (e.g., Qwen MoE / Mixtral-style).
- Focused on pruning only.
- No evaluation harness in this repo.

If you want the full research stack (evaluation + additional compression strategies), use the upstream Cerebras repo.

## Dev

```bash
pnpm verify
pnpm test
pnpm lint
```

## References

- Paper: https://arxiv.org/abs/2510.13999
- Cerebras implementation: https://github.com/CerebrasResearch/reap
- MLX: https://github.com/ml-explore/mlx
