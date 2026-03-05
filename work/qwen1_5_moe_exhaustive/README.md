# Qwen1.5-MoE A2.7B exhaustive REAP run

Current run assumptions:
- model family: `Qwen/Qwen1.5-MoE-A2.7B-Chat` (remote) / `mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit` (local)
- dataset: `roneneldan/TinyStories` materialized to `test-output/tinystories-train10.jsonl`
- sample count: 10
- max tokens per sample: 128
- local methods: `frequency`, `ean_sum`, `weighted_ean_sum`
- remote method: `reap`

Artifacts are written under `test-output/qwen1_5_moe_exhaustive/` locally and under the chosen remote output directory for the upstream run.
