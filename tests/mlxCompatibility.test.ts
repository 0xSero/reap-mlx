import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import { __testOnly as collectorTestOnly } from '../src/core/mlxCollector.js';
import { __testOnly as pruneTestOnly } from '../src/core/mlxPrune.js';

function stripMain(script: string): string {
  return script.replace(/\nif __name__ == '__main__':\n\s+main\(\)\n?$/, '\n');
}

function runPythonHarness(script: string, harness: string): string {
  const result = spawnSync('python3', ['-c', `${stripMain(script)}\n${harness}`], {
    encoding: 'utf8'
  });

  expect(result.status).toBe(0);
  expect(result.stderr).toBe('');
  return result.stdout.trim();
}

describe('mlx embedded python compatibility resolvers', () => {
  it('collector resolves layer paths in the intended order', { timeout: 15_000 }, () => {
    const stdout = runPythonHarness(
      collectorTestOnly.pythonScript(),
      `
class Node:
    pass

top = Node()
top.layers = ['top']

wrapped = Node()
wrapped.language_model = Node()
wrapped.language_model.model = Node()
wrapped.language_model.model.layers = ['language']

mixed = Node()
mixed.layers = ['top']
mixed.language_model = Node()
mixed.language_model.model = Node()
mixed.language_model.model.layers = ['language-preferred']

legacy = Node()
legacy.model = Node()
legacy.model.layers = ['legacy']

assert resolve_layers(top) == ['top']
assert resolve_layers(wrapped) == ['language']
assert resolve_layers(mixed) == ['language-preferred']
assert resolve_layers(legacy) == ['legacy']
print('ok')
`
    );

    expect(stdout).toBe('ok');
  });

  it('collector resolves embed tokens without touching unrelated branches', { timeout: 15_000 }, () => {
    const stdout = runPythonHarness(
      collectorTestOnly.pythonScript(),
      `
class Node:
    pass

def marker(label):
    def inner(x):
        return f"{label}:{x}"
    return inner

wrapped = Node()
wrapped.language_model = Node()
wrapped.language_model.model = Node()
wrapped.language_model.model.embed_tokens = marker('language')
wrapped.vision_model = Node()
wrapped.vision_model.embed_tokens = marker('vision')

legacy = Node()
legacy.model = Node()
legacy.model.embed_tokens = marker('legacy')

top = Node()
top.embed_tokens = marker('top')

assert resolve_embed_tokens(wrapped)('x') == 'language:x'
assert resolve_embed_tokens(legacy)('x') == 'legacy:x'
assert resolve_embed_tokens(top)('x') == 'top:x'
print('ok')
`
    );

    expect(stdout).toBe('ok');
  });

  it('apply resolver accepts top-level, language_model, and legacy layer paths', { timeout: 15_000 }, () => {
    const stdout = runPythonHarness(
      pruneTestOnly.pythonScript(),
      `
class Node:
    pass

top = Node()
top.layers = ['top']

wrapped = Node()
wrapped.language_model = Node()
wrapped.language_model.model = Node()
wrapped.language_model.model.layers = ['language']

mixed = Node()
mixed.layers = ['top']
mixed.language_model = Node()
mixed.language_model.model = Node()
mixed.language_model.model.layers = ['language-preferred']

legacy = Node()
legacy.model = Node()
legacy.model.layers = ['legacy']

assert resolve_layers(top) == ['top']
assert resolve_layers(wrapped) == ['language']
assert resolve_layers(mixed) == ['language-preferred']
assert resolve_layers(legacy) == ['legacy']
print('ok')
`
    );

    expect(stdout).toBe('ok');
  });
});
