#!/usr/bin/env node
// How many zstd decoders a page loads for a site's rest bundle (the lazy
// files as one framed tar, `BUNDLE#OFFSET`; lazy-fetch.mjs). A decoder is a
// wasm instance with a memory of its own, an engine reservation; one in
// each of a desktop's Workers ran the page out of them, and programs then
// failed to start. So:
//
//   a site with no rest bundle: no Worker loads one;
//   a site with one, and the page's lazy-file owner (lazy-cache-worker.js):
//     no Worker loads one, the owner loads exactly one, and every Worker's
//     read of a rest bundle file through the owner gives the file's bytes;
//   a site with one and no owner: a Worker compiles its own, and makes its
//     instance (and memory) only at a first decode.
//
// Workers are worker threads that do what a page Worker does before and
// after its kernel starts (worker.mjs: `setLazyCache`, the rest bundle
// flag; gex.mjs: `lazyReady`; then lazy-fetch.mjs `lazyFetch` reads). The
// owner is the real lazy-cache-worker.js in a worker thread, over a Cache
// Storage on disk, and the bundle comes from a ranged server in another
// process. A decoder load is counted where it happens: each
// WebAssembly.instantiate with zstddec's import object.
//
//   node test/fixtures/page/rest-bundle-decoders.mjs     prints "ok   rest bundle decoders: ..." or FAIL
import { pathToFileURL as gexPathToFileURL } from 'node:url';
import { join as gexJoin } from 'node:path';
// gex: the checkout these fixtures are in, or GEX_ROOT.
const GEX_ROOT = process.env.GEX_ROOT || gexJoin(import.meta.dirname, '..', '..', '..');
const gexUrl = (p) => gexPathToFileURL(gexJoin(GEX_ROOT, p)).href;
import { Worker } from 'node:worker_threads';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
const { makeControl } = await import(gexUrl('hosts/browser/js/lazy-cache.mjs'));

const dir = mkdtempSync(join(tmpdir(), 'rest-bundle-decoders-'));
const site = join(dir, 'site'), store = join(dir, 'cache');
mkdirSync(site, { recursive: true });
mkdirSync(store, { recursive: true });
let server = null;
const cleanup = () => { server?.kill(); rmSync(dir, { recursive: true, force: true }); };
const fail = (m) => { console.log(`FAIL rest bundle decoders: ${m}`); cleanup(); process.exit(1); };

// The bundle: three files' bytes one after another (a tar's place does not
// matter here), cut into 1 MiB frames, each compressed on its own.
const FRAME = 1 << 20;
const fill = (len, seed) => { const b = Buffer.alloc(len); for (let i = 0; i < len; i++) b[i] = ((i * 2654435761 + seed) >>> 13) & 255; return b; };
const files = [fill(FRAME / 2 + 77, 1), fill(FRAME * 2 + 4096, 2), fill(FRAME + 3, 3)];
const offsets = [];
let at = 0;
for (const f of files) { offsets.push(at); at += f.length; }
const plain = Buffer.concat(files);
const frames = [], parts = [];
for (let o = 0, k = 0; o < plain.length; o += FRAME, k++) {
  const p = join(dir, `frame${k}`);
  writeFileSync(p, plain.subarray(o, o + FRAME));
  execFileSync('zstd', ['-q', '-f', '-3', p, '-o', `${p}.zst`]);
  const z = readFileSync(`${p}.zst`);
  frames.push([parts.reduce((a, b) => a + b.length, 0), z.length]);
  parts.push(z);
}
writeFileSync(join(site, 'rest.bin'), Buffer.concat(parts));
writeFileSync(join(site, 'rest.bin.json'), JSON.stringify({ frame: FRAME, size: plain.length, frames }));

server = spawn(process.execPath, ['-e', `
  const http = require('node:http'), fs = require('node:fs'), path = require('node:path');
  const site = ${JSON.stringify(site)};
  http.createServer((req, res) => {
    const p = path.join(site, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    let st; try { st = fs.statSync(p); } catch { res.writeHead(404).end(); return; }
    let a = 0, b = st.size - 1, code = 200; const h = { 'Cache-Control': 'no-store' };
    const m = /^bytes=(\\d+)-(\\d*)$/.exec(req.headers.range || '');
    if (m) { a = +m[1]; if (m[2] !== '') b = Math.min(b, +m[2]); code = 206; h['Content-Range'] = 'bytes ' + a + '-' + b + '/' + st.size; }
    res.writeHead(code, h); fs.createReadStream(p, { start: a, end: b }).pipe(res);
  }).listen(0, '127.0.0.1', function () { console.log(this.address().port); });
`], { stdio: ['ignore', 'pipe', 'inherit'] });
const port = await new Promise((r) => server.stdout.once('data', (d) => r(Number(String(d).trim()))));
const bundle = `http://127.0.0.1:${port}/rest.bin`;

// Counts zstddec's instances in the thread it runs in: the package's own
// asynchronous ones and the late, synchronous one (zstd-late.mjs).
const counter = `
  let decoders = 0, compiled = 0;
  const compile = WebAssembly.compile;
  WebAssembly.compile = function (...args) { compiled++; return compile.apply(WebAssembly, args); };
  const instantiate = WebAssembly.instantiate;
  WebAssembly.instantiate = function (bytes, imports, ...rest) {
    if (imports?.env?.emscripten_notify_memory_growth) decoders++;
    return instantiate.call(WebAssembly, bytes, imports, ...rest);
  };
  WebAssembly.Instance = new Proxy(WebAssembly.Instance, {
    construct(target, args, nt) {
      if (args[1]?.env?.emscripten_notify_memory_growth) decoders++;
      return Reflect.construct(target, args, nt);
    },
  });
`;

// The owner, with \`self\` and a Cache Storage on disk; asked 'count', it
// says how many decoders it loaded.
const ownerSrc = `
  const { parentPort, workerData } = require('node:worker_threads');
  const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
  ${counter}
  const file = (name, key) => path.join(workerData.store, crypto.createHash('sha1').update(name + '\\n' + key).digest('hex'));
  globalThis.caches = { open: async (name) => ({
    match: async (key) => { try { return new Response(fs.readFileSync(file(name, String(key)))); } catch { return undefined; } },
    put: async (key, res) => { fs.writeFileSync(file(name, String(key)), Buffer.from(await res.arrayBuffer())); },
    delete: async (key) => { fs.rmSync(file(name, String(key)), { force: true }); return true; },
  }) };
  globalThis.self = globalThis;
  self.postMessage = (m) => parentPort.postMessage(m);
  import(workerData.worker).then(() => parentPort.on('message', (data) => {
    if (data === 'count') parentPort.postMessage({ kind: 'count', decoders });
    else self.onmessage({ data });
  }));
`;

// A page Worker: its setup, \`lazyReady\` as before its kernel starts, then
// reads of the bundle's files (each [index, from, len]) into the kernel's
// memory at \`dst\`; it answers with its decoder count and what it read.
const readerSrc = `
  const { parentPort, workerData: d } = require('node:worker_threads');
  ${counter}
  (async () => {
    const lc = await import(d.lazyCache), lf = await import(d.lazyFetch);
    if (d.ctl) lc.setLazyCache({ ctl: d.ctl });
    // setRestBundle where the tree has it, else the old global.
    if (d.restBundle) lf.setRestBundle ? lf.setRestBundle(true) : (globalThis.GEX_REST_BUNDLE = 1);
    await lf.lazyReady();
    const read = lf.lazyFetch(d.memory);
    const mem = new Uint8Array(d.memory.buffer);
    const got = [];
    for (const [href, from, len, dst] of d.reads) {
      const url = new TextEncoder().encode(href);
      mem.set(url, d.urlAt);
      got.push(read(d.urlAt, url.length, BigInt(from), dst, len));
    }
    parentPort.postMessage({ decoders, compiled, got });
  })().catch((e) => parentPort.postMessage({ error: String(e?.stack ?? e) }));
`;
const lazyCache = gexUrl('hosts/browser/js/lazy-cache.mjs'), lazyFetch = gexUrl('hosts/browser/js/lazy-fetch.mjs');
const memory = new WebAssembly.Memory({ initial: 256, maximum: 256, shared: true }); // 16 MiB
const runReader = (opts) => new Promise((resolve) => {
  const w = new Worker(readerSrc, { eval: true, workerData: { lazyCache, lazyFetch, memory, reads: [], urlAt: 0, ...opts } });
  w.once('message', (m) => { w.terminate(); resolve(m); });
  w.once('error', (e) => resolve({ error: String(e) }));
});

// 1. No rest bundle, no owner: a Worker loads no decoder.
const none = await runReader({});
if (none.error) fail(`no bundle: ${none.error}`);
if (none.decoders !== 0) fail(`a site with no rest bundle: a Worker loaded ${none.decoders} decoders, not 0`);

// 2. A rest bundle and the owner: three Workers read (at once) spans of
// the three files, across frames; none loads a decoder, the owner one.
const ctl = makeControl();
const owner = new Worker(ownerSrc, { eval: true, workerData: { store, worker: gexUrl('page/src/lazy-cache-worker.js') } });
const ready = await new Promise((r) => {
  owner.on('message', (m) => { if (m.kind === 'ready') r(m); });
  owner.once('error', (e) => r({ ok: false, why: String(e) }));
  owner.postMessage({ ctl, memory, cache: 'gex-pieces', versions: {}, zstd: gexUrl('page/vendor/zstddec/zstddec-stream.mjs') });
});
if (!ready.ok) fail(`owner not ready: ${ready.why}`);
const spans = [[0, 0, files[0].length], [1, 1000, FRAME + 5000], [2, 0, files[2].length], [1, FRAME * 2, 4096]];
const MiB = 1 << 20;
// Each read lands in a place of its own in the 16 MiB memory: the URLs in
// the first MiB, then the three Workers' four spans of at most 1 MiB + 5000.
const plans = [0, 1, 2].map((w) => spans.map(([f, from, len], i) =>
  [`${bundle}#${offsets[f]}`, from, len, MiB + (w * spans.length + i) * (MiB + 8192), f]));
const results = await Promise.all(plans.map((p, w) => runReader({ ctl, restBundle: true, urlAt: w * 4096, reads: p.map(([h, from, len, dst]) => [h, from, len, dst]) })));
const ownerCount = await new Promise((r) => { owner.on('message', (m) => { if (m.kind === 'count') r(m.decoders); }); owner.postMessage('count'); });
await owner.terminate();
results.forEach((res, w) => {
  if (res.error) fail(`Worker ${w}: ${res.error}`);
  if (res.decoders !== 0) fail(`a site with a rest bundle and its owner: Worker ${w} loaded ${res.decoders} decoders, not 0`);
  plans[w].forEach(([, from, len, dst, f], i) => {
    if (res.got[i] !== len) fail(`Worker ${w} read ${res.got[i]} bytes of file ${f} at ${from}, not ${len}`);
    const want = files[f].subarray(from, from + len);
    if (!Buffer.from(new Uint8Array(memory.buffer, dst, len)).equals(want)) fail(`Worker ${w}: file ${f} at ${from}: wrong bytes`);
  });
});
if (ownerCount !== 1) fail(`a site with a rest bundle: the owner loaded ${ownerCount} decoders, not 1`);

// 3. A rest bundle and no owner (no shared Cache Storage): the Worker
// readies its own before its kernel starts, compiled, its instance (and
// memory) left to its first decode (zstd-late.mjs), which a thread that
// never decodes never makes.
const alone = await runReader({ restBundle: true });
if (alone.error) fail(`no owner: ${alone.error}`);
if (alone.compiled !== 1 || alone.decoders !== 0) fail(`a rest bundle with no owner: a Worker compiled ${alone.compiled} decoders and made ${alone.decoders} instances before any decode, not 1 and 0`);

cleanup();
console.log(`ok   rest bundle decoders: none without a rest bundle; with one, 0 in each of ${plans.length} Workers and 1 in the owner, ${plans.length * spans.length} reads right; with no owner 1 compiled in a Worker and no instance before a decode`);
process.exit(0);
