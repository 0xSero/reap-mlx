# reap-mlx

`reap-mlx` is a local, secure CLI package for building REAP-style pruning plans from MLX telemetry.

It reads model expert signals, computes a layer-safe pruning plan, and writes an observation log you can audit after every run.

![Observation engine flow](assets/observation-engine.svg)

## Install

```bash
pnpm install
pnpm build
node dist/cli/index.js help
```

After publishing to npm, the package is intended to run as:

```bash
npx reap-mlx help
```

## Quick start

1) Generate synthetic telemetry

```bash
node dist/cli/index.js init --output ./tmp/telemetry.json --layers 8 --experts 8 --seed 123
```

2) Build pruning plan

```bash
node dist/cli/index.js run --model ./tmp/telemetry.json --output ./tmp/out --ratio 0.4
```

3) Inspect observation log summary

```bash
node dist/cli/index.js observe --file ./tmp/out/observation.log
```

## Commands

- `run`: builds `pruning-plan.json` and `observation.log`
- `observe`: summarizes observation logs from previous runs
- `init`: generates synthetic telemetry input for testing
- `version`, `help`

## Security model

- Blocks path traversal when writing output files
- Refuses symlink reads for model/log input
- Enforces numeric bounds and schema checks on telemetry
- Limits input file size during reads
- Uses atomic file writes for plan and logs
- Keeps one expert per layer as a safety floor

## Output files

Inside your `--output` directory:

- `pruning-plan.json` — structured plan with pruned/kept experts and stats
- `observation.log` — JSONL event stream emitted by the observation engine

## Development

```bash
pnpm lint
pnpm test
pnpm verify
```
