#!/usr/bin/env node
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDefaultReapMlxComponents, runReapMlx } from '../../dist/core/index.js';

function parseArgs(argv) {
  const options = new Map();
  for (let index = 2; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token?.startsWith('--')) continue;
    const key = token.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith('--')) {
      options.set(key, true);
      continue;
    }
    options.set(key, next);
    index += 1;
  }
  return options;
}

function required(options, key) {
  const value = options.get(key);
  if (typeof value !== 'string' || value.trim() === '') throw new Error(`Missing --${key}`);
  return value;
}

async function main() {
  const options = parseArgs(process.argv);
  const telemetryPath = path.resolve(required(options, 'telemetry'));
  const outputDir = path.resolve(required(options, 'output-dir'));
  const ratio = Number(options.get('ratio') ?? '0.25');
  const minExperts = Number(options.get('min-experts') ?? '1');
  const methods = String(options.get('methods') ?? 'frequency,ean_sum,weighted_ean_sum')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);

  await fs.mkdir(outputDir, { recursive: true });

  const components = createDefaultReapMlxComponents();
  const rawTelemetry = await components.loadTelemetry(telemetryPath);
  const telemetry = components.validateTelemetry(rawTelemetry);

  const manifest = [];
  for (const method of methods) {
    const methodDir = path.join(outputDir, method);
    await fs.mkdir(methodDir, { recursive: true });

    const scoreResult = components.scoreSaliency(telemetry.experts, {
      allowLegacySaliency: false,
      pruneMethod: method,
    });
    const selection = components.selectPruning(scoreResult.scoredExperts, {
      targetRatio: ratio,
      minExpertsPerLayer: minExperts,
      preserveSuperExperts: false,
      preserveOutliers: false,
    });
    const decisions = components.buildDecisions(scoreResult.scoredExperts, selection);
    const decisionMap = new Map();
    for (const row of [...decisions.pruned, ...decisions.kept]) {
      decisionMap.set(`${row.layer}:${row.expert}`, row);
    }

    const tracePath = path.join(methodDir, 'decision-trace.jsonl');
    const scoredPath = path.join(methodDir, 'scored-experts.json');
    const summaryPath = path.join(methodDir, 'decision-summary.json');

    const scoredPayload = scoreResult.scoredExperts.map((expert) => {
      const key = `${expert.layer}:${expert.expert}`;
      const decision = decisionMap.get(key);
      return {
        ...expert,
        pruneMethod: method,
        blockedByLayerSafety: selection.blockedKeys.has(key),
        finalReason: decision?.reason ?? 'unknown',
        finalPruned: Boolean(decision && decisions.pruned.some((row) => row.layer === expert.layer && row.expert === expert.expert)),
      };
    });
    await fs.writeFile(scoredPath, `${JSON.stringify(scoredPayload, null, 2)}\n`);

    const traceLines = scoredPayload.map((row) => JSON.stringify(row)).join('\n') + '\n';
    await fs.writeFile(tracePath, traceLines);

    const plan = await runReapMlx({
      modelPath: telemetryPath,
      outputDir: methodDir,
      targetRatio: ratio,
      minExpertsPerLayer: minExperts,
      pruneMethod: method,
      allowLegacySaliency: false,
    });

    const summary = {
      method,
      telemetryPath,
      planPath: path.join(methodDir, 'pruning-plan.json'),
      tracePath,
      scoredPath,
      requestedPruneCount: selection.requestedPruneCount,
      targetPruneCount: selection.targetPruneCount,
      threshold: selection.threshold,
      blockedByLayerSafety: selection.blockedByLayerSafety,
      prunedExperts: plan.stats.prunedExperts,
      keptExperts: plan.stats.keptExperts,
      legacyFallbackUsed: plan.legacyFallbackUsed,
    };
    await fs.writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
    manifest.push(summary);
  }

  const manifestPath = path.join(outputDir, 'manifest.json');
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(JSON.stringify({ outputDir, manifestPath, methods }, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
