import { spawnSync, type SpawnSyncReturns } from 'node:child_process';
import path from 'node:path';

export interface MlxCoherenceProbeConfig {
  modelPath: string;
  prompt: string;
  pythonBin?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface MlxCoherenceProbeResult {
  modelPath: string;
  prompt: string;
  completion: string;
}

function pythonScript(): string {
  return `import argparse
import json
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-tokens', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.2)
    args = parser.parse_args()

    model, tokenizer = load(args.model, lazy=True)
    sampler = make_sampler(temp=float(args.temperature))
    output = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=int(args.max_tokens),
        sampler=sampler,
        verbose=False,
    )

    payload = {
        'modelPath': args.model,
        'prompt': args.prompt,
        'completion': output,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
`;
}

function throwIfFailed(result: SpawnSyncReturns<string>, commandPreview: string): void {
  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    const stderr = result.stderr?.trim() ?? '';
    const stdout = result.stdout?.trim() ?? '';
    throw new Error(
      `MLX coherence probe failed (exit=${result.status})\n${commandPreview}\n${stderr || stdout}`
    );
  }
}

export async function probeMlxModelCoherence(
  config: MlxCoherenceProbeConfig
): Promise<MlxCoherenceProbeResult> {
  const pythonBin = config.pythonBin && config.pythonBin.trim().length > 0 ? config.pythonBin : 'python3';
  const modelPath = path.resolve(config.modelPath);
  const maxTokens = typeof config.maxTokens === 'number' ? config.maxTokens : 80;
  const temperature = typeof config.temperature === 'number' ? config.temperature : 0.2;

  const args = [
    '-c',
    pythonScript(),
    '--model',
    modelPath,
    '--prompt',
    config.prompt,
    '--max-tokens',
    String(maxTokens),
    '--temperature',
    String(temperature)
  ];

  const commandPreview = `${pythonBin} ${args.join(' ')}`;
  const result = spawnSync(pythonBin, args, {
    encoding: 'utf8',
    maxBuffer: 20 * 1024 * 1024
  });

  throwIfFailed(result, commandPreview);

  const line = result.stdout.trim().split(/\r?\n/).at(-1);
  if (!line) {
    throw new Error('MLX coherence probe produced no JSON result');
  }

  const payload = JSON.parse(line) as Record<string, unknown>;
  return {
    modelPath: String(payload.modelPath ?? modelPath),
    prompt: String(payload.prompt ?? config.prompt),
    completion: String(payload.completion ?? '')
  };
}
