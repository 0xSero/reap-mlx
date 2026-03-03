# How to use reap-mlx

This guide shows a complete local workflow.

## 1) Build the CLI

```bash
pnpm install
pnpm build
```

## 2) Create telemetry input

Use `init` for a synthetic dataset:

```bash
node dist/cli/index.js init \
  --output ./examples/telemetry.json \
  --model-name mixtral-like \
  --layers 16 \
  --experts 8 \
  --seed 2026
```

Or provide your own telemetry JSON in this shape:

```json
{
  "modelName": "your-model",
  "experts": [
    { "layer": 0, "expert": 0, "activationScore": 0.31, "tokenCount": 924 },
    { "layer": 0, "expert": 1, "activationScore": 0.95, "tokenCount": 1804 }
  ]
}
```

## 3) Run pruning planner

```bash
node dist/cli/index.js run \
  --model ./examples/telemetry.json \
  --output ./examples/out \
  --ratio 0.35 \
  --calibration 3
```

Optional flags:

- `--job-id <id>`: set your own job id
- `--observation <filename>`: custom log name under output dir
- `--json`: emit full plan JSON to stdout

## 4) Read observation summary

```bash
node dist/cli/index.js observe --file ./examples/out/observation.log --json
```

You get totals for event count, malformed lines, level counts, stage counts, and stage durations.

## 5) Validate package before release

```bash
pnpm verify
```

## Notes

- `--ratio` is capped at `0.95`
- `--calibration` range is `1..25`
- The planner always preserves at least one expert per layer
