/**
 * Do files dropped on the page land in the guest's filesystem, tree and bytes
 * and modes, and does an aborted drop take back what it wrote?
 *
 * handle.drop (page/src/app.js) reads a drop's tree in the page (drop.js
 * `walk`) and has drop-worker.js write it through its own kernel instance.
 * This runs both against the kernel module in Node: `walk` over a folder as
 * a FileSystemDirectoryHandle and as a webkitGetAsEntry entry gives, and the
 * Worker's code in this thread (its `self` a stand-in), on a kernel stack
 * slice of its own. It checks:
 *   - the copy makes /tmp/drops/1, then /tmp/drops/2, with every folder and
 *     file, the bytes read back the same (one file over the write chunk);
 *   - files are 0644, a script ("#!") 0755, folders 0755;
 *   - each file's modification time is its File's lastModified (to the
 *     nanosecond), each folder's the newest of the files under it, an empty one's the drop's;
 *   - progress ends at {done = total, filesDone = files};
 *   - an aborted drop rejects as aborted and leaves no /tmp/drops/N;
 *   - a move into a folder that has a name already numbers it ("a (2).txt"),
 *     and so does a drop into one, or a name the drop has twice; neither
 *     writes over a file there, and an aborted drop into it removes only
 *     what it made;
 *   - a file is written under a hidden name and renamed to its own when
 *     whole (a watcher of the folder never sees it part written).
 *
 * Usage: node test/fixtures/page/drop.mjs [gex_kernel.x86.wasm]
 */
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pathToFileURL as gexPathToFileURL } from 'node:url';
import { join as gexJoin } from 'node:path';
// gex: the checkout these fixtures are in, or GEX_ROOT.
const GEX_ROOT = process.env.GEX_ROOT || gexJoin(import.meta.dirname, '..', '..', '..');
const gexUrl = (p) => gexPathToFileURL(gexJoin(GEX_ROOT, p)).href;

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = GEX_ROOT;
// The kernel the checkout built: the default build's, else the x86 one.
const built = ['gex_kernel.wasm', 'gex_kernel.x86.wasm'].map((f) => join(ROOT, 'page', 'public', 'gex', f));
const path = process.argv[2] || built.find((f) => existsSync(f)) || built[0];
const module = await WebAssembly.compile(await readFile(path));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 65536, shared: true });
const proc = new WebAssembly.Memory({ initial: 16, maximum: 65535, shared: true });
const env = {
  memory,
  process: proc,
  lazy_fetch: () => 0, block_io: () => -1, page_fs: () => -5,
  process64: new WebAssembly.Memory({ address: 'i64', initial: 0n, maximum: 0n, shared: true }),
};
// The page's own instance, which reads back what the Worker wrote.
const k = (await WebAssembly.instantiate(module, { env })).exports;
const stackTop = (k.k_stack_area() >>> 0) + 3 * (k.k_stack_size() >>> 0);

let failed = 0;
const check = (ok, what) => {
  if (!ok) {
    failed++;
    console.log(`check-drop: FAIL ${what}`);
  }
};

const enc = new TextEncoder();
function put(p) {
  const b = enc.encode(p);
  new Uint8Array(proc.buffer).set(b, 0);
  return b.length;
}
function meta(p) {
  const n = put(p);
  if (k.k_fs_path_meta(0n, n, 4096n) < 0) return null;
  const v = new DataView(proc.buffer, 4096, 72);
  return {
    type: v.getUint32(0, true), mode: v.getUint32(4, true) & 0o7777, size: Number(v.getBigUint64(8, true)),
    mtime: Number(v.getBigInt64(24, true)), mtimeNs: v.getUint32(36, true),
  };
}
function read(p) {
  const h = k.k_fs_open(0n, put(p), 0n, 0);
  if (h < 0) return null;
  const parts = [];
  for (let off = 0; ;) {
    const n = k.k_fs_read(h, BigInt(off), 65536n, 1 << 16);
    if (n <= 0) break;
    parts.push(new Uint8Array(proc.buffer).slice(65536, 65536 + n));
    off += n;
  }
  k.k_fs_close(h);
  return Buffer.concat(parts);
}

// The drop: a folder (d1 with a file and a subfolder), a script, and a
// file over the Worker's 1 MiB write chunk.
const big = new Uint8Array(3 * (1 << 20) + 123);
for (let i = 0; i < big.length; i++) big[i] = (i * 2654435761) >>> 24;
const tree = {
  d1: { 'hello.txt': 'hello-from-host\n', sub: { 'n.txt': 'n\n' }, empty: {} },
  'run.sh': '#!/bin/sh\necho hi\n',
  'big.bin': big,
};
// Each file's lastModified (Unix ms), as a browser gives a dropped file's.
const TIMES = { 'hello.txt': 1700000000123, 'n.txt': 1710000000456, 'run.sh': 1690000000789, 'big.bin': 1600000000000 };
const file = (name, body) => new File([body], name, { lastModified: TIMES[name] ?? 1650000000000 });
// As Chromium's getAsFileSystemHandle gives a folder...
const handleOf = (name, body) =>
  body instanceof Uint8Array || typeof body === 'string'
    ? { kind: 'file', name, getFile: async () => file(name, body) }
    : {
        kind: 'directory',
        name,
        async *entries() {
          for (const [n, b] of Object.entries(body)) yield [n, handleOf(n, b)];
        },
      };
// ...and as webkitGetAsEntry does, its entries in batches.
const entryOf = (name, body) =>
  body instanceof Uint8Array || typeof body === 'string'
    ? { isFile: true, name, file: (ok) => ok(file(name, body)) }
    : {
        isFile: false,
        name,
        createReader() {
          const all = Object.entries(body).map(([n, b]) => entryOf(n, b));
          return { readEntries: (ok) => ok(all.splice(0, 2)) };
        },
      };

const { walk } = await import(gexUrl('page/src/drop.js'));
const byHandle = await walk(Object.entries(tree).map(([n, b]) => ({ handle: Promise.resolve(handleOf(n, b)), entry: null, file: null })));
const byEntry = await walk(Object.entries(tree).map(([n, b]) => ({ handle: null, entry: entryOf(n, b), file: null })));
const paths = (l) => l.map((e) => `${e.path}${e.file ? '' : '/'}`).join(' ');
check(paths(byHandle) === 'd1/ d1/hello.txt d1/sub/ d1/sub/n.txt d1/empty/ run.sh big.bin', `walk by handle: ${paths(byHandle)}`);
check(paths(byEntry) === paths(byHandle), `walk by entry: ${paths(byEntry)}`);

// The Worker, in this thread.
const out = [];
let wake = null;
globalThis.self = { postMessage: (m) => { out.push(m); wake?.(); } };
await import(gexUrl('page/src/drop-worker.js'));
const send = (m) => self.onmessage({ data: m });
const answer = async (id) => {
  for (;;) {
    const m = out.find((x) => x.id === id && !x.progress);
    if (m) return m;
    await new Promise((r) => { wake = r; });
  }
};
send({ spec: { module, memory }, stackTop });

const before = Math.floor(Date.now() / 1000);
send({ id: 1, dir: null, root: '/tmp/drops', entries: byHandle });
const a = await answer(1);
const after = Math.floor(Date.now() / 1000);
check(a.done?.path === '/tmp/drops/1', `first drop: ${JSON.stringify(a)}`);
const last = out.filter((m) => m.id === 1 && m.progress).at(-1)?.progress;
check(last && last.done === last.total && last.total === big.length + 16 + 2 + 18 && last.filesDone === 4 && last.files === 4,
  `progress at the end: ${JSON.stringify(last)}`);
const want = [
  ['d1', 1, 0o755], ['d1/sub', 1, 0o755], ['d1/empty', 1, 0o755],
  ['d1/hello.txt', 0, 0o644], ['d1/sub/n.txt', 0, 0o644], ['run.sh', 0, 0o755], ['big.bin', 0, 0o644],
];
for (const [p, type, mode] of want) {
  const m = meta(`/tmp/drops/1/${p}`);
  check(m && m.type === type && m.mode === mode, `${p}: ${JSON.stringify(m)}, want type ${type} mode ${mode.toString(8)}`);
}
check(read('/tmp/drops/1/d1/hello.txt')?.toString() === 'hello-from-host\n', 'hello.txt bytes');
// Times: files their File's, folders the newest under them, an empty one the drop's.
const secs = (ms) => Math.floor(ms / 1000);
const times = [
  ['d1/hello.txt', secs(TIMES['hello.txt'])], ['d1/sub/n.txt', secs(TIMES['n.txt'])], ['run.sh', secs(TIMES['run.sh'])],
  ['big.bin', secs(TIMES['big.bin'])], ['d1/sub', secs(TIMES['n.txt'])], ['d1', secs(TIMES['n.txt'])], ['', secs(TIMES['n.txt'])],
];
for (const [p, t] of times) {
  const m = meta(`/tmp/drops/1/${p}`.replace(/\/$/, ''));
  check(m && m.mtime === t, `${p || '/tmp/drops/1'} mtime: ${m?.mtime}, want ${t}`);
}
check(meta('/tmp/drops/1/d1/hello.txt')?.mtimeNs === 123e6, `hello.txt mtime ns: ${meta('/tmp/drops/1/d1/hello.txt')?.mtimeNs}`);
const em = meta('/tmp/drops/1/d1/empty')?.mtime;
check(em >= before && em <= after, `an empty folder's mtime is the drop's: ${em}, want ${before}..${after}`);
check(Buffer.compare(read('/tmp/drops/1/big.bin') ?? Buffer.alloc(0), Buffer.from(big)) === 0, 'big.bin bytes');

send({ id: 2, dir: null, root: '/tmp/drops', entries: byEntry.slice(0, 2) });
check((await answer(2)).done?.path === '/tmp/drops/2', 'second drop is /tmp/drops/2');

// Aborted before its bytes are in: nothing left.
send({ id: 3, dir: null, root: '/tmp/drops', entries: byHandle });
send({ abort: 3 });
const c = await answer(3);
check(c.error && c.aborted, `aborted drop: ${JSON.stringify(c)}`);
check(meta('/tmp/drops/3') === null, 'an aborted drop leaves no /tmp/drops/3');

// Moved beside a name already there.
send({ id: 4, move: { from: '/tmp/drops/2', names: ['d1'], into: '/home/u/Desktop' } });
check((await answer(4)).placed?.paths?.[0] === '/home/u/Desktop/d1', 'move into a new folder');
check(meta('/home/u/Desktop/d1/hello.txt')?.mtime === secs(TIMES['hello.txt']), 'a moved file keeps its time');
send({ id: 5, dir: null, root: '/tmp/drops', entries: [{ path: 'd1' }, { path: 'a.txt', file: file('a.txt', 'a') }] });
const e = await answer(5);
send({ id: 6, move: { from: e.done.path, names: ['d1', 'a.txt'], into: '/home/u/Desktop' } });
send({ id: 7, dir: null, root: '/tmp/drops', entries: [{ path: 'a.txt', file: file('a.txt', 'b') }] });
const g = await answer(7);
send({ id: 8, move: { from: g.done.path, names: ['a.txt'], into: '/home/u/Desktop' } });
const f = await answer(6);
const h = await answer(8);
check(JSON.stringify(f.placed?.paths) === '["/home/u/Desktop/d1 (2)","/home/u/Desktop/a.txt"]', `numbered move: ${JSON.stringify(f)}`);
check(h.placed?.paths?.[0] === '/home/u/Desktop/a (2).txt' && read('/home/u/Desktop/a (2).txt')?.toString() === 'b', `numbered file: ${JSON.stringify(h)}`);
check(meta(e.done.path) === null, 'a moved drop\'s directory is gone');

// Into a folder that has the names already, one of them twice in the drop.
send({ id: 9, dir: '/home/u/Desktop', root: '/tmp/drops', entries: [
  { path: 'a.txt', file: file('a.txt', 'c') }, { path: 'a.txt', file: file('a.txt', 'd') },
  { path: 'd1' }, { path: 'd1/x.txt', file: file('x.txt', 'x') },
] });
const i = await answer(9);
check(JSON.stringify(i.done?.names) === '["a (3).txt","a (4).txt","d1 (3)"]', `drop into a folder: ${JSON.stringify(i)}`);
check(read('/home/u/Desktop/a.txt')?.toString() === 'a' && read('/home/u/Desktop/a (4).txt')?.toString() === 'd'
  && read('/home/u/Desktop/d1 (3)/x.txt')?.toString() === 'x', 'what was there is kept, the new names hold the drop');
// Aborted there: what was there stays.
send({ id: 10, dir: '/home/u/Desktop', root: '/tmp/drops', entries: [{ path: 'a.txt', file: file('a.txt', 'e') }, ...byHandle] });
send({ abort: 10 });
const j = await answer(10);
check(j.aborted && read('/home/u/Desktop/a.txt')?.toString() === 'a' && meta('/home/u/Desktop/a (5).txt') === null,
  `an aborted drop into a folder: ${JSON.stringify(j)}`);
check(meta('/home/u/Desktop/.a.txt.10.drop-part') === null, 'an aborted drop leaves no part-written file');
// An abort for a drop that has ended is dropped, not kept for a later one.
send({ abort: 9 });

// A file is written under a hidden name and has its own only when whole:
// between its chunks (a File read slowly), the name is not there yet.
const seen = [];
const slow = {
  name: 'slow.exe', size: 3, lastModified: 1650000000000,
  stream() {
    let n = 0;
    return new ReadableStream({
      pull(c) {
        seen.push([meta('/home/u/Desktop/slow.exe') !== null, meta('/home/u/Desktop/.slow.exe.11.drop-part') !== null]);
        if (n === 3) c.close();
        else c.enqueue(new Uint8Array([77 + n++]));
      },
    });
  },
};
send({ id: 11, dir: '/home/u/Desktop', root: '/tmp/drops', entries: [{ path: 'slow.exe', file: slow }] });
const l = await answer(11);
check(l.done && read('/home/u/Desktop/slow.exe')?.toString() === 'MNO', `a slowly read drop: ${JSON.stringify(l)}`);
check(seen.length >= 3 && seen.slice(1).every(([whole, part]) => !whole && part), `its name only once whole: ${JSON.stringify(seen)}`);
check(meta('/home/u/Desktop/.slow.exe.11.drop-part') === null, 'and the hidden name gone');

if (failed) process.exit(1);
console.log('check-drop: ok (two drops by handle and by entry, modes, times, progress, abort, numbered moves and drops into a folder, renamed when whole)');
