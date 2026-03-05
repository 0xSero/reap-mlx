# How to use reap-mlx

End-to-end walkthrough: build the tool, prune a model, inspect the results.

## Build

```bash
pnpm install
pnpm build
```

## Prune a real model

Step 1: Collect routing telemetry. This runs a forward pass and records how each expert behaves.

```bash
node dist/cli/index.js collect \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --output ./tmp \
  --prompt "Explain sparse MoE routing in one sentence." \
  --max-tokens 64 \
  --layers 0-3 \
  --layer-wise \
  --batch-size 128
```

Use `--layer-wise` when you want each selected layer scored from its own replayed hidden state path. Use `--batch-size` when the token-by-expert scoring step needs a lower peak memory footprint.

Step 2: Build a pruning plan. The ratio is how many experts to prune per layer (0.05 = 5%).

```bash
node dist/cli/index.js run \
  --model ./tmp/telemetry-*.json \
  --output ./tmp/plan \
  --ratio 0.05 \
  --min-experts 1 \
  --no-legacy
```

Step 3: Apply the plan to the checkpoint. Use `--dry-run` first to sanity-check.

```bash
node dist/cli/index.js apply \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --plan ./tmp/plan/pruning-plan.json \
  --output ./tmp/pruned-model \
  --dry-run
```

If the dry run looks right, run it again without `--dry-run` to write the pruned model.

## Test without a model

Generate synthetic telemetry:

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

## Inspect the observation log

Every `run` writes a JSONL log with timing data for each pipeline stage.

```bash
node dist/cli/index.js observe --file ./examples/out/observation.log
```

Add `--json` for machine-readable output.

## Run the test suite

```bash
pnpm verify
```
