import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { delimiter, join } from 'node:path';

export const root = fileURLToPath(new URL('../../', import.meta.url));
export const requireWebsite = createRequire(join(root, 'website/package.json'));
export const outputDir = join(root, 'website/public/assets/videos');
export const cacheDir = join(root, 'scripts/video/.cache');
export function findBinary(name) {
  const candidates = [process.env[name.toUpperCase()], join(root, 'scripts/video/.local', name),
    ...(process.env.PATH || '').split(delimiter).map(dir => join(dir, name))].filter(Boolean);
  for (const binary of candidates) {
    if (!existsSync(binary)) continue;
    try { execFileSync(binary, ['-version'], { stdio: 'ignore' }); return binary; } catch { /* Try next local binary. */ }
  }
  throw new Error(`No working ${name}; set ${name.toUpperCase()} to a local executable.`);
}
export function probe(path) {
  return JSON.parse(execFileSync(findBinary('ffprobe'), ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', path]));
}
