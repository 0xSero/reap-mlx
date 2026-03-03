import { randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { writeTextAtomicSafe } from './security.js';
import type {
  ObservationEvent,
  ObservationLevel,
  ObservationStage,
  ObservationSummary
} from './types.js';

interface ObservationEngineOptions {
  sinkFilePath?: string;
  maxEvents?: number;
  jobId?: string;
}

export class ObservationEngine {
  private readonly events: ObservationEvent[] = [];
  private readonly sinkFilePath: string | undefined;
  private readonly maxEvents: number;
  private readonly jobId: string;

  constructor(options: ObservationEngineOptions = {}) {
    this.sinkFilePath = options.sinkFilePath;
    this.maxEvents = options.maxEvents ?? 2_500;
    this.jobId = options.jobId ?? randomUUID();
  }

  getJobId(): string {
    return this.jobId;
  }

  getEvents(): readonly ObservationEvent[] {
    return this.events;
  }

  record(
    stage: ObservationStage,
    message: string,
    options: {
      level?: ObservationLevel;
      durationMs?: number;
      progressPercent?: number;
      data?: Record<string, unknown>;
    } = {}
  ): ObservationEvent {
    if (this.events.length >= this.maxEvents) {
      this.events.shift();
    }

    const event: ObservationEvent = {
      id: randomUUID(),
      jobId: this.jobId,
      timestamp: new Date().toISOString(),
      stage,
      level: options.level ?? 'info',
      message,
      ...(typeof options.durationMs === 'number' ? { durationMs: options.durationMs } : {}),
      ...(typeof options.progressPercent === 'number'
        ? { progressPercent: options.progressPercent }
        : {}),
      ...(options.data ? { data: options.data } : {})
    };

    this.events.push(event);
    return event;
  }

  async track<T>(
    stage: ObservationStage,
    message: string,
    action: () => Promise<T>
  ): Promise<T> {
    const startedAt = performance.now();

    try {
      const result = await action();
      const durationMs = performance.now() - startedAt;
      this.record(stage, message, { durationMs });
      return result;
    } catch (error) {
      const durationMs = performance.now() - startedAt;
      this.record(stage, `${message} failed`, {
        level: 'error',
        durationMs,
        data: {
          error: error instanceof Error ? error.message : String(error)
        }
      });
      throw error;
    }
  }

  async flush(): Promise<void> {
    if (!this.sinkFilePath) {
      return;
    }

    const lines = this.events.map((event) => JSON.stringify(event)).join('\n');
    const output = lines.length > 0 ? `${lines}\n` : '';
    await writeTextAtomicSafe(this.sinkFilePath, output);
  }

  summary(): ObservationSummary {
    const levels: ObservationSummary['levels'] = {
      info: 0,
      warn: 0,
      error: 0
    };

    const stageCounts: ObservationSummary['stageCounts'] = {};
    const stageDurationsMs: ObservationSummary['stageDurationsMs'] = {};

    for (const event of this.events) {
      levels[event.level] += 1;
      stageCounts[event.stage] = (stageCounts[event.stage] ?? 0) + 1;

      if (typeof event.durationMs === 'number') {
        stageDurationsMs[event.stage] =
          (stageDurationsMs[event.stage] ?? 0) + event.durationMs;
      }
    }

    const startedAt = this.events[0]?.timestamp;
    const endedAt = this.events[this.events.length - 1]?.timestamp;

    const totalDurationMs =
      startedAt && endedAt
        ? Math.max(0, Date.parse(endedAt) - Date.parse(startedAt))
        : 0;

    return {
      jobId: this.jobId,
      totalEvents: this.events.length,
      malformedLines: 0,
      ...(startedAt ? { startedAt } : {}),
      ...(endedAt ? { endedAt } : {}),
      totalDurationMs,
      levels,
      stageCounts,
      stageDurationsMs
    };
  }
}
