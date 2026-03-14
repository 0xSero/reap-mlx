# How to use reap-mlx

End-to-end walkthrough: build the tool, prune a model, inspect the results.

## Build

```bash
pnpm install
pnpm build
```

## Prune a real model

Before you copy a random large-sample recipe from chat, anchor on the published REAP calibration standards:

- `<= 110B` models: **1,024 random samples**, packed to **2,048 tokens**
- `>= 110B` models: **12,228 samples**, maximum sequence length **16,384**, no packing
- use **domain-specific calibration data** for generative tasks instead of defaulting to general corpora like `c4`

For coding models, the paper-backed default dataset is:

- `theblackcat102/evol-codealpaca-v1`

Step 1: Collect routing telemetry. This runs a forward pass and records how each expert behaves.

```bash
node dist/cli/index.js collect \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --output ./tmp \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --dataset-split train \
  --max-samples 1024 \
  --min-samples 1024 \
  --max-tokens 2048 \
  --pack-samples \
  --renorm-topk \
  --layers 0-3 \
  --collect-mode reload_per_layer \
  --batch-size 128 \
  --lazy-load
```

What those flags do:
- `--dataset` pulls a Hugging Face dataset directly. Use this for paper-aligned runs like `evol-codealpaca-v1`.
- `--dataset-file` accepts local `json`, `jsonl`, `csv`, `parquet`, or plain text calibration data.
- `--dataset-text-field` selects the field path for plain text samples. If omitted, the collector tries common fields automatically.
- `--dataset-messages-field` is for multi-turn chat rows; it renders each sample independently via the tokenizer chat template when available.
- `--pack-samples` is the closest local equivalent to the paper's packed `2,048`-token calibration workflow for `<= 110B` models.
- `--renorm-topk` keeps router-weight handling aligned with the current upstream REAP behavior when the model expects top-k renormalization.
- `--batch-size` lowers peak memory inside expert scoring by chunking flattened token activations.
- `--collect-mode reload_per_layer` is the lowest-memory mode available in this repo. It re-runs one selected layer at a time and merges the telemetry afterward.
- `--lazy-load` asks MLX to defer parameter materialization during load.

If you are calibrating for creative writing or math instead of coding, follow the same structure and swap the dataset:

- creative writing: `euclaise/WritingPrompts_curated`
- math: `allenai/tulu-3-sft-personas-math`

For very large tool-use or agentic models, the paper also mixes in:

- `Salesforce/xlam-function-calling-60k`
- `SWE-bench/SWE-smith-trajectories`

Do **not** assume `20k` is the default because it sounds more serious. The paper-backed baseline is `1,024` for the standard `<= 110B` path; anything larger should be justified by an eval sweep.

If you only want replay-per-layer without a fresh model load per selected layer, use:

```bash
--layer-wise
```

or equivalently:

```bash
--collect-mode replay_per_layer
```

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

## Calibration source rules

Use exactly one of:
- `--prompt`
- `--dataset`
- `--dataset-file`

Important knobs:
- `--max-samples`: hard cap on usable samples consumed
- `--min-samples`: fail if the collector cannot find enough usable samples
- `--max-tokens`: per-sample or per-packed-sequence token cap
- `--sample-batch-size`: how many sequences are processed together in one model batch

The collector reports:
- scanned samples
- processed samples
- skipped samples
- model tokens
- packed sequences
- sample batches

so you can verify that calibration is doing what you think it is doing.

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

## Exact prune parity check

When you want to prove that two runs will prune the exact same experts, compare the two telemetry files under one shared prune config:

```bash
node dist/cli/index.js parity \
  --left ./examples/telemetry.json \
  --right ./examples/telemetry.json \
  --output ./examples/parity \
  --prune-method reap \
  --n-experts-to-prune-per-layer 2 \
  --min-experts 1 \
  --no-legacy \
  --require-identical-telemetry
```

This writes:
- `./examples/parity/parity-report.json`
- `./examples/parity/parity-report.md`

and fails if either:
- normalized telemetry differs, or
- the pruned expert set differs.

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
