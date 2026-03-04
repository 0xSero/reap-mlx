# How to use reap-mlx

Complete local workflow from build to pruned model.

## Build

```bash
pnpm install
pnpm build
```

## Option A: Real model

Collect telemetry from a local MLX MoE model:

```bash
node dist/cli/index.js collect \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --output ./tmp \
  --prompt "Explain sparse MoE routing in one sentence." \
  --max-tokens 64 \
  --layers 0-3
```

Build pruning plan from that telemetry:

```bash
node dist/cli/index.js run \
  --model ./tmp/telemetry-*.json \
  --output ./tmp/plan \
  --ratio 0.05 \
  --min-experts 1 \
  --no-legacy
```

Apply the plan to the checkpoint:

```bash
node dist/cli/index.js apply \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --plan ./tmp/plan/pruning-plan.json \
  --output ./tmp/pruned-model
```

Validate before writing (dry run):

```bash
node dist/cli/index.js apply \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --plan ./tmp/plan/pruning-plan.json \
  --output ./tmp/pruned-model \
  --dry-run
```

## Option B: Synthetic telemetry (no model needed)

Generate fake telemetry for testing:

```bash
node dist/cli/index.js init \
  --output ./examples/telemetry.json \
  --model-name mixtral-like \
  --layers 16 \
  --experts 8 \
  --seed 2026
```

Run the planner on it:

```bash
node dist/cli/index.js run \
  --model ./examples/telemetry.json \
  --output ./examples/out \
  --ratio 0.35 \
  --calibration 3
```

## Inspect observation log

Every `run` writes a JSONL log. Summarize it:

```bash
node dist/cli/index.js observe --file ./examples/out/observation.log
```

Add `--json` for machine-readable output.

## Verify everything works

```bash
pnpm verify
```
