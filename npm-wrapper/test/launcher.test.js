'use strict';
// Local executable/HTTP fixtures only; these are not registry-install evidence.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const http = require('node:http');
const { spawnSync, spawn } = require('node:child_process');
const launcher = path.resolve(__dirname, '../bin/semantic-memory-mcp.js');
// The legacy module runs on import; do not exercise its unverified network path.
const api = fs.readFileSync(launcher, 'utf8').includes('module.exports') ? require(launcher) : {};
const body = '#!/usr/bin/env node\nif(process.argv[2]==="--version") console.log("semantic-memory-mcp 9.8.7"); else if(process.argv[2]==="signal") process.kill(process.pid,"SIGTERM"); else process.exit(Number(process.argv[2]||0));\n';
const digest = crypto.createHash('sha256').update(body).digest('hex');
function dir(t) { const p = fs.mkdtempSync(path.join(os.tmpdir(), 'sm-launcher-')); t.after(() => fs.rmSync(p,{recursive:true,force:true})); return p; }
function fixture(t) { const p=path.join(dir(t),'fixture'); fs.writeFileSync(p,body,{mode:0o700}); return p; }
function env(p, overrides={}) { return {...process.env, SEMANTIC_MEMORY_MCP_BINARY:p, SEMANTIC_MEMORY_MCP_SHA256:digest, SEMANTIC_MEMORY_MCP_VERSION:'9.8.7', ...overrides}; }
async function server(t, handler) { const s=http.createServer(handler); await new Promise(r=>s.listen(0,'127.0.0.1',r)); t.after(()=>new Promise(r=>s.close(r))); return `http://127.0.0.1:${s.address().port}/binary`; }
function options(cacheDir,url) { return {cacheDir, platform:'fixture', assets:{fixture:{url,sha256:digest,version:'9.8.7'}},allowHttp:true,env:{}}; }
test('exports resolver without starting downloads on import',()=>assert.equal(typeof api.resolveBinary,'function'));
test('unknown PATH and unsupported platforms fail closed without Cargo',async t=>{ const p=fixture(t); await assert.rejects(()=>api.resolveBinary({platform:'absent',assets:{},env:{PATH:path.dirname(p)},cacheDir:dir(t)}),/No verified release/); });
test('explicit admission requires path plus digest plus exact version',async t=>{ const p=fixture(t); await assert.rejects(()=>api.resolveBinary({env:env(p,{SEMANTIC_MEMORY_MCP_SHA256:''})}),/SHA256/); });
test('explicit digest mismatch fails before execution',async t=>{ const p=fixture(t); await assert.rejects(()=>api.resolveBinary({env:env(p,{SEMANTIC_MEMORY_MCP_SHA256:'0'.repeat(64)})}),/digest/); });
test('explicit version mismatch fails closed',async t=>{ const p=fixture(t); await assert.rejects(()=>api.resolveBinary({env:env(p,{SEMANTIC_MEMORY_MCP_VERSION:'1.0.0'})}),/version/); });
test('relative paths cannot be admitted',async t=>{ await assert.rejects(()=>api.resolveBinary({env:env('fixture')}),/absolute/); });
test('symlinks cannot be admitted',async t=>{ const p=fixture(t), link=path.join(dir(t),'link'); fs.symlinkSync(p,link); await assert.rejects(()=>api.resolveBinary({env:env(link)}),/regular file/); });
test('HTTP error never creates a cache executable or partial file',async t=>{ const cache=dir(t),url=await server(t,(_q,r)=>{r.writeHead(404);r.end(body);}); await assert.rejects(()=>api.resolveBinary(options(cache,url)),/HTTP 404/); assert.deepEqual(fs.readdirSync(cache),[]); });
test('download digest mismatch never creates cache or partial file',async t=>{ const cache=dir(t),url=await server(t,(_q,r)=>r.end('bad')); await assert.rejects(()=>api.resolveBinary(options(cache,url)),/digest/); assert.deepEqual(fs.readdirSync(cache),[]); });
test('download version mismatch never promotes verified bytes',async t=>{ const cache=dir(t),url=await server(t,(_q,r)=>r.end(body)), opts=options(cache,url); opts.assets.fixture.version='1.0.0'; await assert.rejects(()=>api.resolveBinary(opts),/version/); assert.deepEqual(fs.readdirSync(cache),[]); });
test('valid download is atomic and cache is verified on every use',async t=>{ const cache=dir(t); let requests=0; const url=await server(t,(_q,r)=>{requests++;r.end(body);}),opts=options(cache,url); const p=await api.resolveBinary(opts); assert.equal(fs.readFileSync(p,'utf8'),body); assert.equal(await api.resolveBinary(opts),p); assert.equal(requests,1); fs.writeFileSync(p,'tampered'); await assert.rejects(()=>api.resolveBinary(opts),/digest/); assert.equal(requests,1); assert.deepEqual(fs.readdirSync(cache),['semantic-memory-mcp']); });
test('untrusted HTTP transport is rejected before downloading',async t=>{ const opts=options(dir(t),'http://127.0.0.1:1/binary'); delete opts.allowHttp; await assert.rejects(()=>api.resolveBinary(opts),/HTTPS/); });
test('child nonzero exit is preserved with arguments',t=>{ const p=fixture(t), r=spawnSync(process.execPath,[launcher,'23'],{env:env(p),encoding:'utf8',timeout:10000}); assert.equal(r.status,23,r.stderr); });
test('child signal never becomes successful exit',t=>{ const p=fixture(t),r=spawnSync(process.execPath,[launcher,'signal'],{env:env(p),encoding:'utf8',timeout:10000}); assert.equal(r.signal,'SIGTERM',r.stderr); });
test('wrapper forwards operator termination to child',async t=>{ const p=fixture(t); fs.writeFileSync(p,body.replace('else process.exit(Number(process.argv[2]||0));','else { console.log("ready"); setInterval(()=>{},1000); }')); const hash=crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'); const child=spawn(process.execPath,[launcher,'wait'],{env:env(p,{SEMANTIC_MEMORY_MCP_SHA256:hash}),stdio:['ignore','pipe','pipe']}); t.after(()=>{if(child.exitCode===null && child.signalCode===null) child.kill('SIGKILL');}); const result=await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('termination timeout')),10000); child.stdout.once('data',()=>child.kill('SIGTERM')); child.once('exit',(code,signal)=>{clearTimeout(timer);resolve({code,signal});});}); assert.equal(result.signal,'SIGTERM'); });
