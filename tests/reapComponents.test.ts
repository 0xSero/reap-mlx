import { describe, expect, it } from 'vitest';
import { createDefaultReapMlxComponents } from '../src/core/reapComponents.js';

const components = createDefaultReapMlxComponents();

describe('reap components saliency', () => {
  it('uses weighted activation norm sum as primary REAP saliency', () => {
    const scored = components.scoreSaliency(
      [
        {
          layer: 0,
          expert: 0,
          activeTokenCount: 4,
          weightedActivationNormSum: 4
        },
        {
          layer: 0,
          expert: 1,
          activeTokenCount: 4,
          weightedActivationNormSum: 2
        }
      ],
      { allowLegacySaliency: false }
    );

    expect(scored.legacyFallbackUsed).toBe(false);
    expect(scored.scoredExperts[0]?.expert).toBe(1);
    expect(scored.scoredExperts[0]?.signal).toBe(0.5);
    expect(scored.scoredExperts[0]?.saliencySource).toBe('weighted_activation_sum');
    expect(scored.scoredExperts[1]?.expert).toBe(0);
    expect(scored.scoredExperts[1]?.signal).toBe(1);
  });

  it('falls back to legacy activationScore when allowed', () => {
    const scored = components.scoreSaliency(
      [
        {
          layer: 0,
          expert: 0,
          activationScore: 0.4,
          tokenCount: 10
        }
      ],
      { allowLegacySaliency: true }
    );

    expect(scored.legacyFallbackUsed).toBe(true);
    expect(scored.scoredExperts[0]?.saliencySource).toBe('legacy_activation_score');
    expect(scored.scoredExperts[0]?.signal).toBeGreaterThan(0.4);
  });

  it('throws when only legacy field exists and legacy fallback is disabled', () => {
    expect(() =>
      components.scoreSaliency(
        [
          {
            layer: 0,
            expert: 0,
            activationScore: 0.1,
            tokenCount: 5
          }
        ],
        { allowLegacySaliency: false }
      )
    ).toThrow(/Missing REAP saliency fields/i);
  });
});

describe('reap components planning', () => {
  it('respects minExpertsPerLayer during selection', () => {
    const saliency = components.scoreSaliency(
      [
        { layer: 0, expert: 0, activeTokenCount: 2, weightedActivationNormSum: 0.2 },
        { layer: 0, expert: 1, activeTokenCount: 2, weightedActivationNormSum: 0.6 },
        { layer: 1, expert: 0, activeTokenCount: 2, weightedActivationNormSum: 0.1 },
        { layer: 1, expert: 1, activeTokenCount: 2, weightedActivationNormSum: 0.4 }
      ],
      { allowLegacySaliency: false }
    );

    const selection = components.selectPruning(saliency.scoredExperts, {
      targetRatio: 0.75,
      minExpertsPerLayer: 1
    });

    expect(selection.requestedPruneCount).toBe(2);
    expect(selection.targetPruneCount).toBe(2);
    expect(selection.prunedExperts.length).toBe(2);

    const decisions = components.buildDecisions(saliency.scoredExperts, selection);
    expect(decisions.pruned.length).toBe(2);
    expect(decisions.pruned.every((entry) => entry.reason === 'low_signal_pruned')).toBe(true);
    expect(decisions.kept.every((entry) => entry.reason !== 'low_signal_pruned')).toBe(true);
  });
});
