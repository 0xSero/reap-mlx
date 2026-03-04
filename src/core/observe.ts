import path from 'node:path';
import { readTextFileSafe } from './security.js';
import type {
  ObservationEvent,
  ObservationLevel,
  ObservationStage,
  ObservationSummary
} from './types.js';

function emptyLevels(): Record<ObservationLevel, number> {
  return {
    info: 0,
    warn: 0,
    error: 0
  };
}

function stageFromUnknown(value: unknown): ObservationStage | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const allowedStages = [
    'bootstrap',
    'load_model',
    'validate',
    'score_experts',
    'plan_pruning',
    'write_output',
    'complete',
    'observe'
  ] as const satisfies readonly ObservationStage[];

  const allowed = new Set<ObservationStage>(allowedStages);

  return allowed.has(value as ObservationStage)
    ? (value as ObservationStage)
    : undefined;
}

function levelFromUnknown(value: unknown): ObservationLevel {
  if (value === 'warn' || value === 'error') {
    return value;
  }
  return 'info';
}

function parseLine(rawLine: string): ObservationEvent | undefined {
  const line = rawLine.trim();
  if (line.length === 0) {
    return undefined;
  }

  const parsed = JSON.parse(line) as Partial<ObservationEvent>;
  const stage = stageFromUnknown(parsed.stage);

  if (!stage || typeof parsed.jobId !== 'string' || typeof parsed.timestamp !== 'string') {
    return undefined;
  }

  return {
    id: typeof parsed.id === 'string' ? parsed.id : 'unknown',
    jobId: parsed.jobId,
    timestamp: parsed.timestamp,
    stage,
    level: levelFromUnknown(parsed.level),
    message: typeof parsed.message === 'string' ? parsed.message : '',
    ...(typeof parsed.durationMs === 'number' ? { durationMs: parsed.durationMs } : {}),
    ...(typeof parsed.progressPercent === 'number'
      ? { progressPercent: parsed.progressPercent }
      : {}),
    ...(parsed.data && typeof parsed.data === 'object'
      ? { data: parsed.data as Record<string, unknown> }
      : {})
  };
}

export async function summarizeObservationLog(
  observationFilePath: string
): Promise<ObservationSummary> {
  const fullPath = path.resolve(observationFilePath);
  const lines = (await readTextFileSafe(fullPath)).split(/\r?\n/);

  const levels = emptyLevels();
  const stageCounts: ObservationSummary['stageCounts'] = {};
  const stageDurationsMs: ObservationSummary['stageDurationsMs'] = {};

  let malformedLines = 0;
  let totalEvents = 0;
  let startedAt: string | undefined;
  let endedAt: string | undefined;
  let jobId: string | undefined;

  for (const rawLine of lines) {
    if (rawLine.trim().length === 0) {
      continue;
    }

    let event: ObservationEvent | undefined;

    try {
      event = parseLine(rawLine);
    } catch {
      malformedLines += 1;
      continue;
    }

    if (!event) {
      malformedLines += 1;
      continue;
    }

    totalEvents += 1;
    levels[event.level] += 1;
    stageCounts[event.stage] = (stageCounts[event.stage] ?? 0) + 1;

    if (typeof event.durationMs === 'number') {
      stageDurationsMs[event.stage] =
        (stageDurationsMs[event.stage] ?? 0) + event.durationMs;
    }

    jobId ??= event.jobId;
    startedAt ??= event.timestamp;
    endedAt = event.timestamp;
  }

  const totalDurationMs =
    startedAt && endedAt
      ? Math.max(0, Date.parse(endedAt) - Date.parse(startedAt))
      : 0;

  return {
    ...(jobId ? { jobId } : {}),
    totalEvents,
    malformedLines,
    ...(startedAt ? { startedAt } : {}),
    ...(endedAt ? { endedAt } : {}),
    totalDurationMs,
    levels,
    stageCounts,
    stageDurationsMs
  };
}
