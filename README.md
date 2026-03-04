# reap-mlx

REAP expert pruning for MLX models. Collect routing telemetry from a running model, build a pruning plan using REAP saliency, and structurally remove experts from the checkpoint.

Three commands do the work:

```
collect  →  run  →  apply
```

`collect` runs a forward pass through your MLX MoE model and records per-expert gate values and activation norms. `run` reads that telemetry, scores experts by saliency, and writes a pruning plan. `apply` slices the pruned experts out of the checkpoint and saves a smaller model.

## Quick start

```bash
pnpm install && pnpm build
```

Collect telemetry from a local MLX model:

```bash
node dist/cli/index.js collect \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --output ./tmp \
  --prompt "Explain sparse MoE routing." \
  --layers 0-3
```

Build a pruning plan at 5% ratio:

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

## How saliency works

Each expert gets a score based on how much routing weight it receives and how large its output activations are. The primary metric is:

```
saliency = mean(g_j(x) * ||f_j(x)||)
```

where `g_j(x)` is the gating weight and `||f_j(x)||` is the L2 norm of the expert output, averaged over active tokens. This follows the REAP paper's formulation.

When `weightedActivationNormSum` is missing from telemetry, the scorer falls back to:

1. `averageGateValue * averageActivationNorm`
2. `(gateValueSum / activeTokenCount) * (activationNormSum / activeTokenCount)`
3. `activationScore` (legacy, disabled with `--no-legacy`)

Experts are ranked ascending by saliency. The lowest-scoring ones get pruned.

## Per-layer pruning

Pruning happens per layer, not globally. For each layer:

```
pruneCount = min(
  floor(numExperts * targetRatio),
  numExperts - minExpertsPerLayer
)
```

This keeps at least `--min-experts` (default 1) alive in every layer, so no layer goes completely dead.

## Telemetry format

The collector outputs JSON with this structure:

```json
{
  "modelName": "qwen1.5-moe-a2.7b-chat-4bit",
  "experts": [
    {
      "layer": 0,
      "expert": 3,
      "activeTokenCount": 42,
      "gateValueSum": 9.33,
      "activationNormSum": 18.71,
      "weightedActivationNormSum": 3.91,
      "averageGateValue": 0.2221,
      "averageActivationNorm": 0.4455,
      "activationScore": 0.0931
    }
  ],
  "metadata": {
    "source": "mlx_real_collector_exact",
    "selectedLayers": "0,1,2,3",
    "topK": 4,
    "renormTopK": false
  }
}
```

The minimum fields `run` needs in strict mode (`--no-legacy`) are `layer`, `expert`, `activeTokenCount`, and `weightedActivationNormSum`.

## Commands

### collect

Runs a forward pass through an MLX MoE model and captures per-expert routing statistics.

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

Reads telemetry, scores saliency, and writes a pruning plan.

```
--model <file>        Telemetry JSON from collect
--output <dir>        Output directory for plan + observation log
--ratio <0..0.95>     Target prune ratio
--calibration <1..25> Calibration rounds (default: 2)
--min-experts <n>     Min experts kept per layer (default: 1)
--no-legacy           Require REAP saliency fields, disable activationScore fallback
--json                Print full plan to stdout
```

### apply

Structurally removes pruned experts from an MLX checkpoint.

```
--model <dir>         Source MLX model directory
--plan <file>         Pruning plan JSON from run
--output <dir>        Output pruned model directory
--dry-run             Validate the plan without writing files
```

### observe

Parses the JSONL observation log from a run and prints a summary.

```
--file <path>         Observation log file
--json                Output as JSON
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

## Observation log

Every `run` writes a JSONL observation log alongside the pruning plan. Each line is a timestamped event with a stage label (`bootstrap`, `load_model`, `validate`, `score_experts`, `plan_pruning`, `write_output`, `complete`) and optional duration.

Use `observe` to get aggregated counts and timings.

## REAP parity

The saliency computation matches the Cerebras REAP implementation:

- Per-expert `g(x) * ||f(x)||` averaged over active tokens
- Optional top-k renormalization in the collector
- Per-layer pruning ratios (not global pooling)
- Structural checkpoint patching that slices expert-axis tensors

What's not here yet: the full benchmark/eval harness from the Cerebras repo, and support for architectures beyond the `switch_mlp` pattern used by Qwen MoE and similar models.

## Security

- Output writes are path-traversal-guarded
- Input file reads refuse symlinks
- Plan and log writes use atomic rename
- Numeric inputs are bounds-checked

## Development

```bash
pnpm verify          # lint + build + test
pnpm test            # build + run vitest
pnpm lint            # typecheck only
```
