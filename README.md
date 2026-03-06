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
  --dataset-file ./calibration/tinystories.jsonl \
  --dataset-format jsonl \
  --dataset-text-field text \
  --max-samples 1024 \
  --min-samples 512 \
  --max-tokens 512 \
  --sample-batch-size 8 \
  --pack-samples \
  --collect-mode reload_per_layer \
  --batch-size 128 \
  --lazy-load
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

- `collect`: runs prompt or dataset calibration and saves per-expert routing + activation stats.
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
--dataset-file <path> Local calibration dataset file
--dataset-format <fmt> auto|json|jsonl|csv|parquet|text
--dataset-split     Dataset split (default: train)
--dataset-text-field Field path for text (default: auto common fields)
--dataset-messages-field Field path for chat message arrays
--max-samples <n>   Max dataset samples to aggregate (default: 100)
--min-samples <n>   Require at least n usable samples (default: 1)
--max-tokens <n>    Token cap (default: 256)
--sample-batch-size <n> Batch multiple samples/conversations together
--pack-samples      Pack multiple independent samples into max-tokens windows
--layers <spec>     Example: "0-3,8,10"
--renorm-topk       Renormalize top-k gate weights
--layer-wise        Re-run forward passes per selected layer before scoring
--collect-mode <name> single_pass|replay_per_layer|reload_per_layer
--batch-size <n>    Chunk flattened token activations during scoring
--lazy-load         Ask MLX to lazily materialize weights during load
--python <bin>      Python binary (default: python3)
```

Notes:
- Use exactly one of `--prompt`, `--dataset`, or `--dataset-file`.
- `--dataset-file` supports local `json`, `jsonl`, `csv`, `parquet`, and plain `text`.
- `--dataset-messages-field` renders multi-turn samples through the tokenizer chat template when available.
- `--sample-batch-size` is real multi-sample mini-batching.
- `--pack-samples` packs independent samples to fill context windows more efficiently.
- `--batch-size` is token-chunking inside expert scoring.
- `--collect-mode reload_per_layer` is the lowest-memory observation mode in this repo today: it re-runs one layer at a time and writes a merged telemetry file at the end. It reduces working-set pressure, but it is slower.
- `--lazy-load` asks MLX to defer parameter materialization; use it with `reload_per_layer` when you want the most conservative memory profile.

### full

```text
--model <dir>       MLX model directory
--output <dir>      Pipeline output directory
--prompt <text>     Calibration text
--dataset <name>    HuggingFace dataset name
--dataset-file <path> Local calibration dataset file
--dataset-format <fmt> auto|json|jsonl|csv|parquet|text
--dataset-split     Dataset split (default: train)
--dataset-text-field Field path for text (default: auto common fields)
--dataset-messages-field Field path for chat message arrays
--max-samples <n>   Max dataset samples to aggregate (default: 100)
--min-samples <n>   Require at least n usable samples (default: 1)
--max-tokens <n>    Token cap (default: 256)
--sample-batch-size <n> Batch multiple samples/conversations together
--pack-samples      Pack multiple independent samples into max-tokens windows
--layers <spec>     Example: "0-3,8,10"
--renorm-topk       Renormalize top-k gate weights
--layer-wise        Re-run forward passes per selected layer before scoring
--collect-mode <name> single_pass|replay_per_layer|reload_per_layer
--batch-size <n>    Chunk flattened token activations during scoring
--lazy-load         Ask MLX to lazily materialize weights during load
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

- Supports `switch_mlp`-style MLX MoE checkpoints (e.g., Qwen MoE / Mixtral-style), including full-precision and quantized expert weights in the collector.
- Focused on pruning only.
- The new `reload_per_layer` mode lowers observation memory pressure, but it is still not true disk-streamed per-layer checkpoint loading for apply/writeback.
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
