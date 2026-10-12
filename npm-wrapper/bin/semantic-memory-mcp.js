#!/usr/bin/env node
'use strict';
// Verified selection of a separate Rust executable; no PATH/Cargo fallback.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const https = require('node:https');
const http = require('node:http');
const { spawn, spawnSync } = require('node:child_process');
const NAME = 'semantic-memory-mcp';
const PREFIX = 'SEMANTIC_MEMORY_MCP';
const MAX_BYTES = 128 * 1024 * 1024;
const ASSETS = Object.freeze({
  'linux-x64': Object.freeze({
    url: 'https://github.com/RecursiveIntell/semantic-memory-mcp/releases/download/v1.1.0/semantic-memory-mcp-linux-x64',
    sha256: '5089f22f1e0f3f423b9086b018036c509b74579ff49bc844dab05062b8c629a3',
    version: '0.5.6',
  }),
});
function failure(code, message) { return new Error(`${code}: ${message}`); }
function digest(bytes) { return crypto.createHash('sha256').update(bytes).digest('hex'); }
function admission(hash, version) {
  if (!/^[a-f0-9]{64}$/.test(hash || '')) throw failure('E_ADMISSION_INVALID', 'SHA256 must be an exact lowercase hexadecimal digest');
  if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version || '')) throw failure('E_ADMISSION_INVALID', 'version must be exact, not a tag or range');
}
function verifyBinary(file, hash, version) {
  let stat;
  try { stat = fs.lstatSync(file); } catch { throw failure('E_BINARY_TYPE', 'binary must exist as a regular file'); }
  if (!stat.isFile() || stat.isSymbolicLink()) throw failure('E_BINARY_TYPE', 'binary must be a regular file, not a symlink');
  if (stat.size > MAX_BYTES) throw failure('E_BINARY_SIZE', 'binary exceeds the admitted size bound');
  if (digest(fs.readFileSync(file)) !== hash) throw failure('E_DIGEST', 'binary digest mismatch; refusing execution');
  const probe = spawnSync(file, ['--version'], { encoding: 'utf8', timeout: 5000, maxBuffer: 65536 });
  if (probe.error || probe.signal || probe.status !== 0) throw failure('E_VERSION_PROBE', 'bounded version probe failed');
  if (probe.stdout.trim() !== `${NAME} ${version}`) throw failure('E_VERSION', 'binary version does not match admission');
  if (digest(fs.readFileSync(file)) !== hash) throw failure('E_DIGEST', 'binary changed during the version probe');
  return file;
}
function download(url, allowHttp, deadline = Date.now() + 60000, redirects = 0) {
  return new Promise((resolve, reject) => {
    let address;
    try { address = new URL(url); } catch { reject(failure('E_DOWNLOAD_URL', 'invalid release URL')); return; }
    const fixtureHttp = allowHttp && address.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(address.hostname);
    if ((address.protocol !== 'https:' && !fixtureHttp) || address.username || address.password) {
      reject(failure('E_DOWNLOAD_URL', 'HTTPS without URL credentials is required')); return;
    }
    const remaining = deadline - Date.now();
    if (remaining <= 0 || redirects > 5) { reject(failure('E_DOWNLOAD_BOUND', 'download deadline or redirect bound exceeded')); return; }
    const transport = address.protocol === 'https:' ? https : http;
    let timer;
    const request = transport.get(address, { headers: { 'User-Agent': 'RecursiveIntell-verified-wrapper' } }, response => {
      if ([301, 302, 303, 307, 308].includes(response.statusCode)) {
        response.resume();
        clearTimeout(timer);
        if (!response.headers.location) { reject(failure('E_DOWNLOAD_HTTP', 'redirect lacks a destination')); return; }
        download(new URL(response.headers.location, address).href, allowHttp, deadline, redirects + 1).then(resolve, reject);
        return;
      }
      if (response.statusCode !== 200) {
        response.resume(); clearTimeout(timer);
        reject(failure('E_DOWNLOAD_HTTP', `HTTP ${response.statusCode}`)); return;
      }
      const chunks = []; let bytes = 0;
      response.on('data', chunk => {
        bytes += chunk.length;
        if (bytes > MAX_BYTES) request.destroy(failure('E_DOWNLOAD_SIZE', 'download exceeds size bound'));
        else chunks.push(chunk);
      });
      response.on('aborted', () => reject(failure('E_DOWNLOAD_PARTIAL', 'download was interrupted')));
      response.on('error', reject);
      response.on('end', () => {
        clearTimeout(timer);
        if (!response.complete) reject(failure('E_DOWNLOAD_PARTIAL', 'incomplete response'));
        else resolve(Buffer.concat(chunks));
      });
    });
    timer = setTimeout(() => request.destroy(failure('E_DOWNLOAD_TIMEOUT', 'total download deadline exceeded')), remaining);
    request.setTimeout(15000, () => request.destroy(failure('E_DOWNLOAD_TIMEOUT', 'download idle timeout')));
    request.on('error', error => { clearTimeout(timer); reject(error); });
  });
}
async function resolveBinary(options = {}) {
  const env = options.env || process.env;
  const file = env[`${PREFIX}_BINARY`];
  const hash = env[`${PREFIX}_SHA256`];
  const version = env[`${PREFIX}_VERSION`];
  if (file || hash || version) {
    if (!file || !path.isAbsolute(file)) throw failure('E_ADMISSION_INVALID', 'binary path must be absolute');
    admission(hash, version);
    return verifyBinary(file, hash, version);
  }
  const platform = options.platform || `${os.platform()}-${os.arch()}`;
  const asset = (options.assets || ASSETS)[platform];
  if (!asset) throw failure('E_PLATFORM', 'No verified release for this platform; no automatic PATH or Cargo fallback');
  admission(asset.sha256, asset.version);
  const cache = options.cacheDir || path.join(os.homedir(), '.cache', 'recursiveintell', NAME, platform, asset.version, asset.sha256);
  const target = path.join(cache, NAME);
  if (fs.existsSync(target)) return verifyBinary(target, asset.sha256, asset.version);
  const bytes = await download(asset.url, options.allowHttp === true);
  if (digest(bytes) !== asset.sha256) throw failure('E_DIGEST', 'download digest mismatch');
  fs.mkdirSync(cache, { recursive: true, mode: 0o700 });
  const temporary = path.join(cache, `${NAME}.partial-${crypto.randomUUID()}`);
  try {
    fs.writeFileSync(temporary, bytes, { flag: 'wx', mode: 0o700 });
    verifyBinary(temporary, asset.sha256, asset.version);
    fs.renameSync(temporary, target);
    return verifyBinary(target, asset.sha256, asset.version);
  } finally {
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
  }
}
async function main() {
  const file = await resolveBinary();
  const child = spawn(file, process.argv.slice(2), { stdio: 'inherit', env: process.env });
  const timers = [];
  const forward = signal => {
    child.kill(signal);
    const timer = setTimeout(() => child.kill('SIGKILL'), 5000);
    timer.unref(); timers.push(timer);
  };
  const onTerm = () => forward('SIGTERM');
  const onInt = () => forward('SIGINT');
  process.on('SIGTERM', onTerm); process.on('SIGINT', onInt);
  const retire = () => {
    timers.forEach(clearTimeout);
    process.removeListener('SIGTERM', onTerm); process.removeListener('SIGINT', onInt);
  };
  child.once('error', () => { retire(); process.stderr.write('E_CHILD_SPAWN: admitted child could not start\n'); process.exitCode = 1; });
  child.once('exit', (code, signal) => {
    retire();
    if (signal) {
      process.stderr.write(`E_CHILD_SIGNAL: child terminated by ${signal}\n`);
      process.kill(process.pid, signal);
    } else process.exitCode = Number.isInteger(code) ? code : 1;
  });
}
module.exports = { resolveBinary };
if (require.main === module) main().catch(error => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
