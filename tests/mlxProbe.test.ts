import { describe, expect, it } from 'vitest';
import * as core from '../src/core/index.js';

describe('probe exports', () => {
  it('re-exports probeMlxModelCoherence from the core entrypoint', () => {
    expect(typeof core.probeMlxModelCoherence).toBe('function');
  });
});
