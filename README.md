# reap-mlx

REAP pruning for MLX MoE models on Apple Silicon.

The shortest version:

```text
collect telemetry -> build a pruning plan -> apply the plan
```

This repo ports the pruning side of [Cerebras REAP](https://github.com/CerebrasResearch/reap) to local MLX workflows. It is built for people who want to calibrate on real data, inspect the telemetry, and physically write a smaller MLX checkpoint.

## upstream links

- Paper: https://arxiv.org/abs/2510.13999
- Original Cerebras REAP repo: https://github.com/CerebrasResearch/reap
- Vendored upstream submodule in this repo: `external/cerebras-reap`

The submodule is there so you can diff local MLX work against the upstream research implementation without leaving the repo.

## what it does today

- Collects per-expert telemetry from an MLX MoE model.
- Builds a pruning plan with REAP or simpler scoring rules.
- Applies that plan to an MLX checkpoint.
- Compares two telemetry files under the same prune config with an exact parity report.
- Supports prompt calibration, Hugging Face datasets, and local dataset files.
- Supports lower-memory collection modes, token chunking, sample mini-batching, packing, and chat-style calibration rows.

## what it does not do yet

- It does not stream checkpoint writeback layer-by-layer during `apply`. Collection can run in lower-memory modes, but apply still loads the model normally.
- It does not ship a full benchmark harness in the repo. You still need to run before/after evals yourself.

## quick start

Requirements:
- Apple Silicon Mac
- Node 20+
- pnpm
- Python 3.11+
- `mlx` and `mlx_lm`

Install and build:

```bash
pnpm install
pnpm build
```

### 1) collect telemetry from a JSONL dataset

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

### 2) build a pruning plan

```bash
node dist/cli/index.js run \
  --model ./tmp/telemetry-*.json \
  --output ./tmp/plan \
  --ratio 0.5 \
  --min-experts 1 \
  --no-legacy
```

### 3) apply pruning

```bash
node dist/cli/index.js apply \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --plan ./tmp/plan/pruning-plan.json \
  --output ./tmp/pruned-model
```

Use `--dry-run` on `apply` if you want to validate the plan before writing a new checkpoint.

## command overview

- `collect`: run prompt or dataset calibration and write telemetry JSON.
- `run`: score experts from telemetry and write `pruning-plan.json`.
- `parity`: run the same prune config against two telemetry files and diff the exact prune set.
- `full`: run `collect -> run -> apply` in one command.
- `apply`: remove pruned experts from the MLX checkpoint.
- `observe`: summarize an observation log.
- `init`: generate synthetic telemetry for local testing.

## how calibration works

You can calibrate with exactly one input source per run:

- `--prompt <text>`
- `--dataset <huggingface-name>`
- `--dataset-file <path>`

Local dataset files support:

- `json`
- `jsonl`
- `csv`
- `parquet`
- plain `text`

Useful controls:

- `--max-samples <n>` limits how many rows are processed.
- `--min-samples <n>` fails the run if too many rows were unusable.
- `--max-tokens <n>` limits how much of each sample is processed.
- `--dataset-text-field <field.path>` selects a text field.
- `--dataset-messages-field <field.path>` renders chat-style rows through the tokenizer chat template when available.

That means you are not stuck with one-off prompts. You can hand the collector a real dataset and bound both sample count and per-sample length.

## paper-aligned calibration guidance

If you want `reap-mlx` to track the published REAP paper more closely, use the paper's calibration standards as your baseline instead of inventing larger sample counts by default.

- For models with `<= 110B` parameters, the paper calibrates on **1,024 randomly selected samples** packed to **2,048 tokens**.
- For models with `>= 110B` parameters, the paper uses **12,228 samples** with a maximum sequence length of **16,384 tokens** and no packing.
- The paper shows that **domain-specific calibration matters a lot**. In particular, coding models calibrated on `c4` can degrade badly, while `evol-codealpaca` preserves coding quality much better.

Recommended dataset choices from the paper:

- coding: `theblackcat102/evol-codealpaca-v1`
- creative writing: `euclaise/WritingPrompts_curated`
- math: `allenai/tulu-3-sft-personas-math`
- larger tool-use / agentic mixes: add `Salesforce/xlam-function-calling-60k` and `SWE-bench/SWE-smith-trajectories`

For local Apple Silicon workflows, the most practical paper-aligned baseline is the `<= 110B` recipe:

```bash
node dist/cli/index.js collect \
  --model ./models/qwen1.5-moe-a2.7b-chat-4bit \
  --output ./tmp/paper-coding \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --dataset-split train \
  --max-samples 1024 \
  --min-samples 1024 \
  --max-tokens 2048 \
  --pack-samples \
  --renorm-topk \
  --collect-mode reload_per_layer \
  --batch-size 128 \
  --lazy-load
```

Do **not** treat `20k` as a paper-backed default. If you want to test `20k`, treat it as an experiment and compare it against the paper-backed `1,024` sample baseline on your actual eval suite.

## batching and memory modes

There are three different knobs here, and they are not the same thing.

### sample mini-batching

`--sample-batch-size <n>` batches multiple independent samples or conversations into one model pass.

Use this when the goal is throughput on a larger calibration set.

### dataset packing

`--pack-samples` packs multiple independent short samples into fuller context windows.

Use this when your dataset has lots of short rows and you do not need one-row-per-window isolation.

### token chunking inside collection

`--batch-size <n>` chunks flattened token activations inside expert scoring.

Use this when collection is running hot on memory.

### collection modes

`collect` and `full` support these modes:

- `single_pass`: the original behavior
- `replay_per_layer`: replay hidden states layer-by-layer in one process
- `reload_per_layer`: rerun one selected layer per collector process, then merge telemetry

`reload_per_layer` is the lowest-memory observation mode in this repo today. It is slower, but it cuts working-set pressure enough to make some runs practical on smaller machines.

`--layer-wise` remains as shorthand for the layer replay path.

`--lazy-load` asks MLX to defer parameter materialization during load. Pair it with `reload_per_layer` when you want the most conservative collection profile.

## pruning methods

This project supports:

- `reap`
- `reap_l2`
- `frequency`
- `weighted_frequency_sum`
- `ean_sum`
- `ean_mean`
- `ean_ca`
- `weighted_ean_sum`
- `weighted_ean_sum_l2`
- `max_activations`

Notes:
- `weighted_frequency_sum` maps to weighted routing-frequency totals (`weightedExpertFrequencySum`, with `gateValueSum` fallback for local collector telemetry).
- `ean_ca`, `weighted_ean_sum_l2`, and `reap_l2` can require richer telemetry fields when you ingest data produced outside the local collector.

The REAP-style score used here is:

```text
saliency_j = mean( g_j(x) * ||f_j(x)|| )
```

Where:

- `g_j(x)` is the router softmax weight for expert `j`
- `f_j(x)` is the expert output
- the mean is taken over routed tokens for that expert

## exact prune parity

If you are trying to answer this question:

```text
same exact telemetry + same exact prune config => same exact experts pruned?
```

Use `parity`.

```bash
node dist/cli/index.js parity \
  --left ./left.telemetry.json \
  --right ./right.telemetry.json \
  --output ./tmp/parity \
  --prune-method reap \
  --n-experts-to-prune-per-layer 15 \
  --min-experts 1 \
  --no-legacy \
  --require-identical-telemetry
```

It writes:

- `parity-report.json`
- `parity-report.md`
- left and right pruning outputs under the output directory

The report includes:

- normalized telemetry hashes
- the first differing expert row
- the exact prune-set diff
- per-layer expert deltas

This is the cleanest correctness check in the repo. If telemetry is identical and the prune sets differ, something is wrong.

## minimal CLI reference

### `collect`

```text
--model <dir>                    MLX model directory
--output <dir>                   Telemetry output directory
--prompt <text>                  Single calibration text
--dataset <name>                 HuggingFace dataset name
--dataset-file <path>            Local calibration dataset file
--dataset-format <fmt>           auto|json|jsonl|csv|parquet|text
--dataset-split <name>           Dataset split (default: train)
--dataset-text-field <field>     Text field path
--dataset-messages-field <field> Chat messages array field path
--max-samples <n>                Max dataset samples to aggregate (default: 100)
--min-samples <n>                Require at least n usable samples (default: 1)
--max-tokens <n>                 Per-sample token cap (default: 256)
--sample-batch-size <n>          Batch multiple samples or conversations together
--pack-samples                   Pack short independent samples into fuller windows
--layers <spec>                  Example: 0-3,8,10
--renorm-topk                    Renormalize top-k gate weights to sum to 1
--layer-wise                     Enable layer-wise collection mode
--collect-mode <name>            single_pass|replay_per_layer|reload_per_layer
--batch-size <n>                 Token chunk size for collection batching
--lazy-load                      Ask MLX to lazily materialize weights during load
--python <bin>                   Python binary (default: python3)
```

### `full`

```text
--model <dir>                    MLX model directory
--output <dir>                   Pipeline output directory
--prompt <text>                  Single calibration text
--dataset <name>                 HuggingFace dataset name
--dataset-file <path>            Local calibration dataset file
--dataset-format <fmt>           auto|json|jsonl|csv|parquet|text
--dataset-split <name>           Dataset split (default: train)
--dataset-text-field <field>     Text field path
--dataset-messages-field <field> Chat messages array field path
--max-samples <n>                Max dataset samples to aggregate (default: 100)
--min-samples <n>                Require at least n usable samples (default: 1)
--max-tokens <n>                 Per-sample token cap (default: 256)
--sample-batch-size <n>          Batch multiple samples or conversations together
--pack-samples                   Pack short independent samples into fuller windows
--layers <spec>                  Example: 0-3,8,10
--renorm-topk                    Renormalize top-k gate weights to sum to 1
--layer-wise                     Enable layer-wise collection mode
--collect-mode <name>            single_pass|replay_per_layer|reload_per_layer
--batch-size <n>                 Token chunk size for collection batching
--lazy-load                      Ask MLX to lazily materialize weights during load
--ratio <0..0.95>                Target prune ratio per layer
--min-experts <n>                Minimum experts kept per layer
--dry-run                        Validate apply step without writing pruned model
```

### `run`

```text
--model <file>                   Telemetry JSON from collect
--output <dir>                   Plan and observation output directory
--ratio <0..0.95>                Target prune ratio per layer
--calibration <n>                Calibration rounds (default: 2)
--min-experts <n>                Minimum experts kept per layer
--no-legacy                      Require REAP saliency fields
--json                           Print plan JSON to stdout
```

### `parity`

```text
--left <file>                    Left telemetry JSON
--right <file>                   Right telemetry JSON
--output <dir>                   Output directory for left/right plans and parity report
--ratio <0..0.95>                Target prune ratio per layer
--n-experts-to-prune-per-layer <n>
--prune-method <name>            reap|reap_l2|frequency|weighted_frequency_sum|ean_sum|ean_mean|ean_ca|weighted_ean_sum|weighted_ean_sum_l2|max_activations
--require-identical-telemetry    Fail unless normalized telemetry hashes match exactly
--json                           Print parity report JSON to stdout
```

### `apply`

```text
--model <dir>                    Source MLX model
--plan <file>                    pruning-plan.json
--output <dir>                   Pruned model output
--dry-run                        Validate plan only
```

### `observe`

```text
--file <path>                    Observation log file
--json                           JSON output
```

### `init`

```text
--output <file>                  Synthetic telemetry output
--model-name <name>              Default: synthetic-moe
--layers <n>                     Default: 8
--experts <n>                    Default: 8
--seed <int>                     RNG seed
```

## current scope and limits

- This repo is focused on pruning MLX MoE checkpoints, not on being a full research harness.
- It supports `switch_mlp`-style MLX MoE checkpoints, including full-precision and quantized expert weights in the collector.
- Lower-memory collection is here now. Lower-memory apply is not.
- The repo has planner tests, collector wiring tests, and an exact parity harness.
- The repo still does not include a built-in benchmark suite for pruned versus unpruned models.

If you want a broader research stack, including full evaluation workflows and other compression paths, use the upstream Cerebras repo. A checked-out copy now lives in this repo at `external/cerebras-reap`.

## development

```bash
pnpm lint
pnpm build
pnpm test
pnpm verify
```

## references

- Paper: https://arxiv.org/abs/2510.13999
- Cerebras implementation: https://github.com/CerebrasResearch/reap
- MLX: https://github.com/ml-explore/mlx
