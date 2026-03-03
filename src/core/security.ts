import { constants, promises as fs } from 'node:fs';
import { randomUUID } from 'node:crypto';
import path from 'node:path';

const DEFAULT_MAX_BYTES = 8 * 1024 * 1024;

function isSubPath(baseDir: string, candidatePath: string): boolean {
  return (
    candidatePath === baseDir ||
    candidatePath.startsWith(`${baseDir}${path.sep}`)
  );
}

export function resolveSafePath(baseDir: string, relativePath: string): string {
  const normalizedBase = path.resolve(baseDir);
  const resolved = path.resolve(normalizedBase, relativePath);

  if (!isSubPath(normalizedBase, resolved)) {
    throw new Error(`Path traversal blocked: ${relativePath}`);
  }

  return resolved;
}

export async function ensureSecureDir(dirPath: string): Promise<void> {
  await fs.mkdir(dirPath, { recursive: true, mode: 0o700 });
  try {
    await fs.chmod(dirPath, 0o700);
  } catch {
    // Best effort on platforms where chmod semantics vary.
  }
}

async function assertRegularFile(filePath: string): Promise<void> {
  const stats = await fs.lstat(filePath);

  if (stats.isSymbolicLink()) {
    throw new Error(`Refusing to read symlink: ${filePath}`);
  }

  if (!stats.isFile()) {
    throw new Error(`Expected a regular file: ${filePath}`);
  }
}

export async function readTextFileSafe(
  filePath: string,
  maxBytes = DEFAULT_MAX_BYTES
): Promise<string> {
  await assertRegularFile(filePath);
  const stats = await fs.stat(filePath);

  if (stats.size > maxBytes) {
    throw new Error(`File too large (${stats.size} bytes): ${filePath}`);
  }

  return fs.readFile(filePath, 'utf8');
}

export async function readJsonFileSafe<T>(
  filePath: string,
  maxBytes = DEFAULT_MAX_BYTES
): Promise<T> {
  const text = await readTextFileSafe(filePath, maxBytes);

  try {
    return JSON.parse(text) as T;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown JSON parse error';
    throw new Error(`Invalid JSON in ${filePath}: ${message}`);
  }
}

export async function writeTextAtomicSafe(
  filePath: string,
  content: string
): Promise<void> {
  const dir = path.dirname(filePath);
  await ensureSecureDir(dir);

  const tempPath = path.join(dir, `.tmp-${path.basename(filePath)}-${randomUUID()}`);
  const handle = await fs.open(
    tempPath,
    constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY,
    0o600
  );

  try {
    await handle.writeFile(content, 'utf8');
    await handle.sync();
  } finally {
    await handle.close();
  }

  await fs.rename(tempPath, filePath);

  try {
    await fs.chmod(filePath, 0o600);
  } catch {
    // Best effort on platforms where chmod semantics vary.
  }
}

export async function writeJsonAtomicSafe(
  filePath: string,
  payload: unknown
): Promise<void> {
  const body = `${JSON.stringify(payload, null, 2)}\n`;
  await writeTextAtomicSafe(filePath, body);
}

function parseNumericInput(value: unknown, label: string): number {
  if (typeof value === 'number') {
    return value;
  }

  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }

  throw new Error(`Expected numeric value for ${label}`);
}

export function assertFiniteNumber(
  value: unknown,
  label: string,
  min?: number,
  max?: number
): number {
  const parsed = parseNumericInput(value, label);

  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be finite`);
  }

  if (typeof min === 'number' && parsed < min) {
    throw new Error(`${label} must be >= ${min}`);
  }

  if (typeof max === 'number' && parsed > max) {
    throw new Error(`${label} must be <= ${max}`);
  }

  return parsed;
}

export function assertInteger(
  value: unknown,
  label: string,
  min?: number,
  max?: number
): number {
  const parsed = assertFiniteNumber(value, label, min, max);

  if (!Number.isInteger(parsed)) {
    throw new Error(`${label} must be an integer`);
  }

  return parsed;
}
