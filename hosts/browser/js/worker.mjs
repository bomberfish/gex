// One guest process, in its own Worker, with its own address space. The
// Worker's side of things; `host.mjs` is the other half.
//
// `Gex` builds its own `WebAssembly.Memory`, so a Worker has a private address
// space -- nothing collides, and a process that blocks blocks its own Worker.
// What processes share -- the process table, filesystem, pipes -- is in the
// global kernel, a second wasm module every Worker instantiates against one
// shared memory, so a syscall stays a call rather than a message back here.
//
// Two ways in:
//
//   `run`     load a program. This is pid 1.
//   `adopt`   become a child some other Worker's guest forked, from the
//             record that fork produced.
//
// The same file runs under a browser Worker and under Node's `worker_threads`,
// because Node is the fast lane -- a browser cycle is twenty minutes, Node two.

/** The one difference between a Web Worker and a Node one, hidden here. */
async function channel() {
  if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
    return {
      // `transfer` is optional: `postMessage(m, [])` is legal but
      // `postMessage(m, undefined)` is not, in either host.
      post: (m, transfer) =>
        transfer && transfer.length ? self.postMessage(m, transfer) : self.postMessage(m),
      on: (f) => {
        self.onmessage = (e) => f(e.data);
      },
    };
  }
  const { parentPort } = await import('node:worker_threads');
  let handler = null;
  return {
    post: (m, transfer) =>
      transfer && transfer.length
        ? parentPort.postMessage(m, transfer)
        : parentPort.postMessage(m),
    // One listener, whose handler a given-back Worker's next process
    // replaces, as `onmessage` is replaced in a browser.
    on: (f) => {
      if (!handler) parentPort.on('message', (m) => handler(m));
      handler = f;
    },
  };
}

/// The fork record's bulk: what the parent hands its child (modules, memory
/// image, maps) and the host never reads. In a browser it goes from the
/// parent's Worker to the child's over a channel of its own (`bulk`), so the
/// page's thread deserializes none of it: a module or memory arriving there
/// counts against its isolate's external memory, and at a fork's thousands of
/// modules that ran one full collection after another on the thread every
/// process waits on. The single-thread build and Node keep the one message.
const FORK_BULK = ['modules', 'image', 'vmaBlob', 'pages', 'shared'];
const forkDirect = () => !(SP_CODE && single.on) && typeof MessageChannel === 'function' && typeof WorkerGlobalScope !== 'undefined';
/** How far a record's image reaches (host.mjs `imageExtent`), said in the
 *  record that travels without it. */
function forkExtent(child) {
  let n = 0;
  for (const r of child?.image ?? []) n = Math.max(n, r.end ?? 0, r.at + (r.len ?? r.bytes?.byteLength ?? 0));
  return n;
}
/** A fork record's bulk, when it came apart from the record: awaited from
 *  the parent's Worker and put back. */
async function forkBulkIn(child) {
  const p = child?.bulk;
  if (!p) return;
  delete child.bulk;
  const bulk = await new Promise((res) => {
    p.onmessage = (e) => res(e.data);
  });
  p.close();
  Object.assign(child, bulk);
}

// Static, not a run-time specifier: a bundler cannot resolve `import(msg.url)`,
// so the built Worker chunk 404'd on a bare `./gex.mjs` and rejected where
// nobody listened. This file and `gex.mjs` are siblings everywhere.
import { setFill, fillFile, lazyReady, kernelThread, lazyFetch, processMemories } from './lazy-fetch.mjs';
import { blockIo } from './blocks.mjs';
import { lateZstd } from './zstd-late.mjs';
import { forkTransfer } from './fork-transfer.mjs';
import { setLazyCache } from './lazy-cache.mjs';
import { pageFs, setPageFs } from './page-fs.mjs';
// The host's clock object, looked up once in this realm: `performance` is a
// Blink attribute of the global scope, and reading it costs more than `now()`.
const perf = globalThis.performance;

import { noteFetch, noteStatus, setNoteSink } from './fetch-note.mjs';
import { workerOptions } from './options.mjs';
import { Gex, Env, utf8, narrate, hostNowNs, vforkTrace } from './gex.mjs';
import { single, registerWorkerMain, sched } from './single.mjs';
import { SP_CODE } from './mode.mjs';
import { logNamed, setLog } from './log.mjs';
// GEX_WATCH_DIR=<page>: narrate every write to that page's directory entry.
if (typeof process !== 'undefined' && process.env?.GEX_WATCH_DIR) {
  Gex.watchDir = Number(process.env.GEX_WATCH_DIR);
}

// Under a block trace at node, keep the trace (the shim's `debug` channel,
// console.debug -> stdout) on the same stream as the syscall log and the crash
// report (stderr): two streams interleave arbitrarily and the stdout tail is
// lost when the process is killed, which hides the very blocks before a crash.
if (typeof process !== 'undefined' && (process.env?.GEX_TRACE_BLOCKS || process.env?.GEX_SYNC_CONSOLE)) {
  // Synchronous, too: a Worker's console goes through the main thread
  // asynchronously, and the megabytes a crashing process traces just before
  // it is killed never arrive -- the one stretch the trace is for. Not an
  // await: a bundler's classic-script Worker cannot hold one at the top.
  const fs = process.getBuiltinModule('node:fs');
  const sync = (...a) => {
    try { fs.writeSync(2, a.map((x) => (typeof x === 'string' ? x : String(x))).join(' ') + '\n'); } catch {}
  };
  console.error = sync;
  console.debug = sync;
}

// What a process leaves in this realm besides `workerMain`'s state, which the
// next process on this Worker must not see (`reset`): its timers, and the
// globals its start message set. Not in the single-thread build, whose
// Workers are fibers of one realm and are never given back.
const timers = new Set();
const realmGlobals = new Set(Object.getOwnPropertyNames(globalThis));
if (!(SP_CODE && single.on)) {
  for (const [make, clear, once] of [['setTimeout', 'clearTimeout', true], ['setInterval', 'clearInterval', false]]) {
    const mk = globalThis[make];
    const cl = globalThis[clear];
    if (typeof mk !== 'function' || typeof cl !== 'function') continue;
    globalThis[make] = function (f, ...rest) {
      let id;
      const g = once && typeof f === 'function' ? function (...a) { timers.delete(id); return f.apply(this, a); } : f;
      id = mk.call(globalThis, g, ...rest);
      timers.add(id);
      return id;
    };
    globalThis[clear] = function (id) {
      timers.delete(id);
      return cl.call(globalThis, id);
    };
  }
}
/** This realm as a new Worker has it, before the next process: every timer
 *  the last one left stopped, every global it set gone. The console's
 *  marking (`__gexMarkedConsole`) stays, since `console.error` keeps it. */
function resetRealm() {
  for (const id of timers) {
    clearTimeout(id);
    clearInterval(id);
  }
  timers.clear();
  for (const k of Object.getOwnPropertyNames(globalThis)) {
    if (realmGlobals.has(k) || k === '__gexMarkedConsole') continue;
    try { delete globalThis[k]; } catch {}
  }
}

// Everything below is one Worker's: its port, its process, its state. A
// function rather than module scope so the single-thread build (single.mjs)
// can run many of these in one realm, one per fiber; a real Worker runs one.
//
// And so a Worker can be given back (`reset`, host.mjs `#giveBack`): a
// process that exited cleanly leaves this function's state behind, and the
// next process starts on a new call of it, in a realm whose JS is compiled
// and warm. `carry` is what outlives a process: the page's modules, given
// while it waited in the pool. Messages the old call still posts are dropped.
async function workerMain(rawPort, carry = null) {
const life = { on: true };
// The host's word on the other memory flavour's runtime (host.mjs
// `#otherGate`), which an exec of a program of that flavour waits on
// (gex.mjs `other_runtime`), and how to ask for it.
function otherGate(msg) {
  globalThis.GEX_OTHER_GATE = msg.otherGate ? new Int32Array(msg.otherGate) : undefined;
  globalThis.GEX_ASK_OTHER = msg.otherGate ? () => port.post({ kind: 'loadOther' }) : undefined;
  // The single-thread build's, which has no shared gate (host.mjs `#otherLoadFn`).
  globalThis.GEX_LOAD_OTHER = msg.otherLoad;
}

const port = {
  post: (m, transfer) => {
    if (life.on) rawPort.post(m, transfer);
  },
  on: rawPort.on,
};
// What this Worker fetches and installs, to its host (fetch-note.mjs). Not
// in the single-thread build, whose fibers share the page's realm (and the
// last fiber's port would outlive none of them).
if (!(SP_CODE && single.on)) setNoteSink((note) => port.post({ kind: 'note', note }));


// This Worker's runtime, once its start message builds it. Module-scope so a
// `sharedModule` message (relayed sibling-Worker compiles) can reach it.
let workerGex = null;
/// The unit cache's small files the page's Workers read before this one
/// started (host.mjs `#unitFiles`), by path; null for one the cache lacks.
const unitFiles = new Map();
/// The largest unit-cache file a Worker shares with the page that way.
const UNIT_FILE_MAX = 4 << 20;
// Which guest thread this Worker runs, asked for by a profiler or
// debugger (`Runtime.evaluate` in the Worker's session): a pooled Worker is
// made before it has a process, so its name cannot say. Read when asked,
// so it follows an `execve`.
globalThis.__gexWho = () => workerGex?.who ?? null;
/// The page's modules, given while this Worker waited in the host's pool
/// (`pooledModules`): its runtime starts with them, and the start message
/// carries only those it lacks.
const pooled = carry?.pooled ?? new Map();
/// Which process this is of the ones this Worker has run, from 1.
const runs = (carry?.runs ?? 0) + 1;
/// The bridges every process of a page sets up (GL, Wayland, network), as
/// modules once loaded. Imported while this Worker waits in the host's pool
/// (the end of this function), and used from here without an `await import`
/// when they have arrived: an import of a module already evaluated still
/// waits for a turn of this Worker's event loop, which was 1.5 ms of every
/// fork child's start on the demo page.
const bridgeImports = {
  gl: () => import('./gl-bridge.mjs'),
  wl: () => import('./wl-bridge.mjs'),
  net: () => import('./net-bridge.mjs'),
};
const bridges = {};
const bridge = (k) => bridges[k] ?? bridgeImports[k]().then((m) => (bridges[k] = m));
/// This Worker's runtime made ahead of its process (`preattach`, host.mjs):
/// a promise of `{ gex, shared, stack, mem64 }`, or null. Used by an exec'd
/// posix_spawn child, which runs on a fresh memory; anything else leaves it.
let preattached = null;
/// Whether this process's memory can go to the next fork child when it exits
/// (`Gex#recycleMemory`): not once another Worker holds it -- a thread, the
/// compositor reading its pixels, the session reaching its door.
let recyclable = true;
/// A spare's thread (`Gex#adoptThread`): the host's `spareGo`, held until
/// the spare waits for it, or the wait it answers. `thread` null: let go.
let spareGot = null;
let spareWaiter = null;

// Standard input, as it arrives. A guest read that finds nothing asks the
// host and waits here; `readStdin` below is what the run loop awaits.
let stdinWaiter = null;
let stdinEof = false;
// `lflag` reads the guest's termios through `gex_tty_lflag`, which takes the
// kernel -- so it may only be called from outside the shim. Calling it from the
// `write` import (inside a syscall, kernel already borrowed) took the borrow
// twice and panicked. It is read when the guest parks for input and remembered
// for the writes in between.
let lflag = () => 0;
let lastLflag = 0;
let winsize = () => {};

// Nothing in here may fail quietly: the handler is async, so a throw becomes a
// rejected promise nobody holds, and a Worker that rejects on the way in looks
// exactly like one that is busy.
port.on((msg) => {
  // A spare's thread, or its letting go (`spareGot`): never a start.
  if (msg?.kind === 'spareGo') {
    if (spareWaiter) spareWaiter(msg.thread ?? null);
    else spareGot = msg;
    spareWaiter = null;
    return;
  }
  // Given back to the pool: this process's state stays here, unreachable,
  // and the next one starts on a new call (see `workerMain`).
  if (msg?.kind === 'reset') {
    life.on = false;
    resetRealm();
    workerMain(rawPort, { pooled, runs }).catch((e) => {
      console.error('gex: a given-back worker could not start again:', e?.stack ?? e);
      rawPort.post({ kind: 'threw', message: String(e?.message ?? e) });
    });
    return;
  }
  handle(msg).catch((e) => {
    const what = e?.stack ?? e?.message ?? String(e);
    if (typeof console !== 'undefined') console.error('gex: worker failed:', what);
    port.post({ kind: 'threw', message: what });
  });
});

/**
 * The guest's hottest back edges, as lines of `count  where`. Read here
 * because naming an address needs this thread's shim and memory.
 */
function hotBlocks(gex) {
  const e = gex.shim.exports;
  if (!e.gex_hot_blocks) return [];
  const at = e.gex_alloc(4096) >>> 0;
  const n = e.gex_hot_blocks(at, 4096) >>> 0;
  const text = utf8(new Uint8Array(gex.memory.buffer, at, n)).trim();
  if (!text) return [];
  return text.split('\n').map((line) => {
    const [pc, count] = line.split(' ');
    // An address alone is a library and an offset nobody has worked out yet.
    const nameAt = e.gex_alloc(160) >>> 0;
    const len = e.gex_object_at ? e.gex_object_at(BigInt(pc), nameAt, 160) >>> 0 : 0;
    const where = len
      ? utf8(new Uint8Array(gex.memory.buffer, nameAt, len))
      : pc;
    return `${String(count).padStart(10)}  ${where}`;
  });
}

// Closure libraries arrive in their own messages -- all of them in the start
// message will not cross a `postMessage` (Chromium `DataCloneError, out of
// memory`). Collected here and installed by `load` before the guest runs:
// installing them after it starts left libraries without slots and the shell
// never printed a prompt.
const arrived = [];
let expected = 0;
let allHere = null;
let allHereIs = null;

function noteClosureModule(entry) {
  arrived.push(entry);
  if (allHereIs && arrived.length >= expected) allHereIs();
}

/// Compiled libraries this Worker has asked the host for, by index.
const wanted = new Map();

/// Ask the host for closure library `i`, and answer when it lands. The module
/// is fetched only for the branch that lands in it; registration needs only
/// the metadata. Sending all modules up front cost 549ms per fork.
function wantLibrary(i) {
  let w = wanted.get(i);
  if (!w) {
    let settle;
    w = { p: new Promise((r) => { settle = r; }), settle };
    wanted.set(i, w);
    port.post({ kind: 'wantLibrary', i });
  }
  return w.p;
}

/** Every closure library this Worker was promised, once they are all in. */
function closureModules(n) {
  expected = n;
  if (!n || arrived.length >= n) return Promise.resolve(arrived);
  allHere = allHere ?? new Promise((r) => { allHereIs = r; });
  if (arrived.length >= expected) allHereIs();
  return allHere.then(() => arrived);
}

/**
 * A process that never got a runtime (no memory could be made for it) ends
 * in the kernel as one killed would: its status for its parent's `wait4`
 * (a `vfork` parent's wait with it), its pipe and terminal ends let go. The
 * global kernel is instantiated for it here on the kernel spec's spare
 * memories, so this makes no memory of its own.
 */
async function endUnstarted(msg) {
  const spec = msg.kernel, pid = msg.child?.pid ?? msg.pid;
  if (!spec?.module || !spec.memory || !pid || !spec.spare) return;
  const mod = spec.module instanceof WebAssembly.Module ? spec.module : await WebAssembly.compile(spec.module);
  const inst = await WebAssembly.instantiate(mod, {
    env: { memory: spec.memory, ...processMemories(spec.spare.m32, false, spec.spare), lazy_fetch: lazyFetch(spec.memory), block_io: blockIo(spec.memory), page_fs: pageFs(spec.memory) },
  });
  const k = inst.exports;
  if (msg.kernelStack && k.__stack_pointer) kernelThread(k, msg.kernelStack);
  if (k.k_proc_no_memory) k.k_proc_no_memory(pid);
  else k.k_exit(pid, -9);
  k.k_pipe_forget?.(pid);
  k.k_pty_forget?.(pid);
}

async function handle(msg) {
  // GEX_LOG from the page (`extra.log`): a Worker sees neither its globals nor its query.
  setLog(msg?.log);
  if (msg?.unitFiles?.buf) {
    const { buf, index } = msg.unitFiles;
    for (const [rel, off, len] of index) if (!unitFiles.has(rel)) unitFiles.set(rel, len < 0 ? null : new Uint8Array(buf, off, len));
  }
  if (msg?.kind === 'closureMeta') {
    // Registration only: which library owns which page. The compiled module
    // for each is asked for when a branch lands in it.
    (msg.metas ?? []).forEach((meta, i) =>
      noteClosureModule({ meta, mod: null, program: (msg.programs ?? [])[i] ?? '' }));
    expected = arrived.length;
    allHereIs?.();
    return;
  }
  if (msg?.kind === 'closureModule') {
    // The answer to a `wantLibrary`.
    if (typeof msg.i === 'number') {
      const w = wanted.get(msg.i);
      if (arrived[msg.i]) arrived[msg.i].mod = msg.mod;
      w?.settle(msg.mod);
      return;
    }
    noteClosureModule(msg.entry);
    return;
  }
  if (msg?.kind === 'stdin') {
    stdinWaiter?.(msg.bytes);
    stdinWaiter = null;
    return;
  }
  if (msg?.kind === 'winsize') {
    winsize(msg.rows, msg.cols);
    return;
  }
  if (msg?.kind === 'eof') {
    stdinEof = true;
    stdinWaiter?.(null);
    stdinWaiter = null;
    return;
  }
  // A sibling Worker's compiled module, relayed by the host so this Worker's
  // `#catchUp` reuses it. Arrives independently of the run/adopt/thread start,
  // and only matters once this Worker has a `Gex`; before that it is dropped and
  // the module is simply recompiled. See `Gex#sharedModules`.
  if (msg?.kind === 'sharedModule') {
    workerGex?.addSharedModule(msg.key, msg.mod);
    return;
  }
  // A sibling's dead modules (`Gex#reapDead`), or the page's units the host
  // let go of: dropped here as well.
  if (msg?.kind === 'droppedModules') {
    workerGex?.dropSharedModules(msg.keys);
    for (const k of msg.keys ?? []) pooled.delete(k);
    return;
  }
  if (msg?.kind === 'pooledModules') {
    for (const [key, mod] of msg.entries ?? []) pooled.set(key, mod);
    return;
  }
  // While this Worker waits in the pool: the kernel, shim and guest
  // instances on a fresh memory, which `attach` would otherwise make after
  // the process arrived (3.3 ms of an exec'd child's 10.6 ms start).
  if (msg?.kind === 'preattach') {
    const shared = !!msg.shared;
    const stack = msg.kernelStack;
    // The memory's index type is the process's, and the host says which: a
    // wasm64 shim imports an i64 memory. Decided here, before the memory is
    // made, since no start message has said it yet (it was left i32, and
    // every m64 Worker's runtime failed to instantiate).
    const mem64 = !!msg.mem64;
    // Either way: a Worker the pool took back may have run the other
    // flavour's process last (a 64-bit program's).
    // Not in the single-thread build, whose realm holds processes of both
    // flavours and the page's own runtime: each Gex is told its own.
    if (!(SP_CODE && single.on)) {
      if (mem64) globalThis.GEX_MEM64 = 1;
      else delete globalThis.GEX_MEM64;
    }
    if (msg.pageFlavours !== undefined) globalThis.GEX_PAGE_FLAVOURS = msg.pageFlavours;
    otherGate(msg);
    preattached = (async () => {
      const gex = new Gex(new Env(), {
        mem64,
        kernel: { ...msg.kernel, stackTop: stack },
        shared,
        modules: pooled,
        fiber: rawPort.fiber ?? null,
      });
      await gex.preattach(msg.shim, msg.guest);
      return { gex, shared, stack, mem64 };
    })().catch((err) => {
      console.error(`[gex] a pooled Worker could not prepare a runtime: ${err?.message ?? err}`);
      return null;
    });
    return;
  }
  // The wineserver's door, from the host: this Worker's door calls go to it
  // from its next one. Before this Worker has a `Gex`, the start message
  // carries it instead.
  if (msg?.kind === 'door') {
    workerGex?.setDoor(msg.door);
    return;
  }
  if (msg?.kind !== 'run' && msg?.kind !== 'adopt' && msg?.kind !== 'thread') return;
  if (msg.kind === 'adopt') await forkBulkIn(msg.child);
  if (msg.kind === 'thread') await forkBulkIn(msg);

  // The page's switches (options.mjs), onto the globals this Worker's runtime
  // reads: carried in the start message, since the page's globals are another
  // context's. Set before anything reads them (`?adopt=N`).
  workerOptions(msg);
  // Mark this Worker's console errors: Chrome logs every level as INFO, so a
  // fault is indistinguishable from progress once read outside, and a Worker
  // cannot reach the page's reporter. Guarded so it is applied once.
  if (!globalThis.__gexMarkedConsole) {
    globalThis.__gexMarkedConsole = true;
    // Under Node, straight to fd 2: a Worker's console is a stream its event
    // loop drains to the parent, so a process that parks in the shim (wait
    // queues: no park bound brings it back to JS) and then exits
    // lost every line it logged meanwhile -- unitcache-fork's GEX_DENSE_DEBUG
    // adopts among them.
    const fs = globalThis.process?.getBuiltinModule?.('fs');
    const realError = fs
      ? (...a) => {
          const t = a.map((x) => (typeof x === 'string' ? x : x?.stack || String(x))).join(' ') + '\n';
          try { fs.writeSync(2, t); } catch { console.error(...a); }
        }
      : console.error.bind(console);
    // One marker per *line*, not per call: Chrome's `--enable-logging=stderr`
    // writes a console message up to its first newline and drops the rest, so a
    // multi-line diagnostic (a `killed by signal` report with its thread,
    // syscall and futex rings) reached the harness as only its first line and
    // the part naming the fault was silently lost. Splitting here means every
    // line is its own INFO:CONSOLE record, marked and captured. An Error is
    // rendered through its stack so that survives too; a lone single-line call
    // is passed through untouched, objects and all.
    console.error = (...a) => {
      const text = a
        .map((x) => (typeof x === 'string' ? x : x?.stack || String(x)))
        .join(' ');
      if (text.includes('\n')) {
        for (const line of text.split('\n')) realError('[gex-console-error]', line);
      } else {
        realError('[gex-console-error]', ...a);
      }
    };
  }
  // Everything the guest writes goes back to whoever started this, posted
  // rather than accumulated so a long run says what it is doing as it runs.
  // How many output messages this Worker has posted, and how many the host
  // says it has delivered (a shared counter it bumps per message).
  let outputSent = 0;
  const outputAck = msg.outputAck ? new Int32Array(msg.outputAck) : null;
  // The page's order of writes (host.mjs `#inOrder`): a number per write, taken as it happens.
  const outputSeq = msg.outputSeq ? new Int32Array(msg.outputSeq) : null;
  // Unit-cache files the page fetched ahead (`file:<rel>` in preModules).
  const preFiles = new Map((msg.preModules ?? []).filter(([k]) => k.startsWith('file:')).map(([k, v]) => [k.slice(5), v]));
  // `closed` (an ejected page's units archive): the page fetched every unit
  // file there is, so a file not among them is a miss, not a request.
  const preClosed = (msg.preModules ?? []).some(([k]) => k === 'closed');
  // A zstd decoder for a split unit's cold clusters (gex_split) and for
  // units served compressed (`cgen.zst`), made before the guest runs: their
  // loads are synchronous.
  let zstd = null;
  if (msg.cgen?.zstd && typeof XMLHttpRequest === 'function') {
    try {
      const t0 = logNamed('unitstall') ? performance.now() : -1;
      // compiled now, instantiated at its first decode: a Worker that never
      // decodes (most threads) makes no memory for it (zstd-late.mjs)
      zstd = await lateZstd(import(msg.cgen.zstd), new URL('./zstddec.wasm', new URL(msg.cgen.zstd, globalThis.location?.href)).href);
      if (t0 >= 0) console.warn(`[unitzstd] decoder ready in ${(performance.now() - t0).toFixed(1)} ms`);
    } catch (err) {
      console.error(`[gex] no zstd decoder for units (${msg.cgen.zstd}): ${err?.message ?? err}`);
      zstd = null;
    }
  }
  class WorkerEnv extends Env {
    /**
     * Wait until the host has delivered everything this Worker wrote: called
     * by the shim just before its process's exit becomes visible to a
     * waiter, so a parent that `wait`s for a child and then prints cannot
     * overtake the child's last output (they cross different ports).
     * Bounded, so a host that has gone away does not hang the exit.
     */
    drainOutput() {
      if (!outputAck) return;
      const deadline = Date.now() + 10000;
      for (;;) {
        const got = Atomics.load(outputAck, 0);
        if (got >= outputSent) return;
        const left = deadline - Date.now();
        if (left <= 0) {
          console.error(`[gex] exit: the host delivered ${got} of ${outputSent} output messages in 10s; exiting anyway`);
          return;
        }
        Atomics.wait(outputAck, 0, got, Math.min(left, 100));
      }
    }
    write(fd, bytes) {
      // Copied: `bytes` is a view on the memory the guest is still running in.
      if (msg.debugExit) {
        // Straight out as well as posted: a crashing Worker takes its last few
        // messages with it, and those are the ones that say why.
        console.error(utf8(bytes.slice()));
      }
      // A page wants text; a command line wants the bytes: `gzip -c` writes a
      // deflate stream that a UTF-8 round trip would fill with U+FFFD.
      outputSent++;
      const seq = outputSeq ? Atomics.add(outputSeq, 0, 1) : undefined;
      port.post(
        msg.rawOutput
          ? { kind: 'output', fd, bytes: bytes.slice(), lflag: lastLflag, seq }
          : { kind: 'output', fd, text: utf8(bytes.slice()), lflag: lastLflag, seq },
      );
    }
    now() {
      return hostNowNs();
    }
    async readStdin() {
      if (stdinEof) return false;
      const bytes = await new Promise((r) => {
        stdinWaiter = r;
        // Outside the shim, so the kernel is free to be asked.
        lastLflag = lflag();
        port.post({ kind: 'stdin', lflag: lastLflag });
      });
      // `true` rather than bytes: a shared terminal already put the keystrokes
      // in the kernel's input queue this process reads, so pushing them again
      // here would deliver every line twice.
      if (bytes === true) return true;
      if (!bytes) return false;
      gex.pushStdin(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
      return true;
    }
    // The guest forked. Handed up: only the host makes Workers.
    async spawnProcess(child) {
      if (forkDirect()) {
        const ch = new MessageChannel();
        const bulk = {};
        for (const k of FORK_BULK) bulk[k] = child[k];
        const head = { ...child, extent: forkExtent(child), bulk: ch.port2 };
        for (const k of FORK_BULK) delete head[k];
        ch.port1.postMessage(bulk, forkTransfer(bulk));
        port.post({ kind: 'fork', child: head }, [ch.port2]);
        return;
      }
      port.post({ kind: 'fork', child }, forkTransfer(child));
    }
    // A thread carries the memory itself, not a copy: a shared
    // `WebAssembly.Memory` crosses `postMessage` as itself, so the other Worker
    // is on this process's address space.
    // A session snapshot's stop: this Worker's thread parked, its frames
    // unwound into its record (`Gex#snapImports`).
    snapParked(info) {
      port.post({ kind: 'snap-parked', ...info, memory: gex.memory });
    }
    async spawnThread(thread) {
      recyclable = false;
      // The memory to the thread's Worker directly, as a fork's bulk (unless
      // a harness keeps every memory on the page, `?memories=1`).
      if (forkDirect() && !msg.memories && thread.memory) {
        const ch = new MessageChannel();
        ch.port1.postMessage({ memory: thread.memory });
        port.post({ kind: 'thread', thread: { ...thread, memory: undefined, bulk: ch.port2 } }, [ch.port2]);
        return;
      }
      port.post({ kind: 'thread', thread });
    }
    // A Worker made ahead for this program's next thread (`Gex#offerSpare`):
    // the host keeps what it needs to start one, the memory among it, while
    // the process lives.
    offerSpare(spare) {
      recyclable = false;
      port.post({ kind: 'threadSpare', spare });
    }
    // A module this Worker just compiled, up to the host to relay to the sibling
    // Workers of the same process so they reuse it rather than recompiling the
    // same registry bytes. A `WebAssembly.Module` crosses `postMessage` by
    // reference -- no byte copy. See `Gex#sharedModules`.
    shareModule(key, mod) {
      port.post({ kind: 'compiledModule', key, mod });
    }
    // Modules of this process no dispatch reaches any more (`Gex#reapDead`):
    // the host forgets them too, or its relay keeps their code alive.
    dropModules(keys) {
      port.post({ kind: 'droppedModules', keys });
    }
    // The wineserver published its door: up to
    // the host, which hands it to every Worker of the session.
    publishDoor(door) {
      recyclable = false;
      port.post({ kind: 'doorPublish', door });
    }
    // A function-tier unit's files by key (`gex.useCgen`): `msg.cgen.root`, a
    // directory under Node (GEX_CGEN_CACHE's layout), or `msg.cgen.url`, read
    // with a synchronous request -- allowed in a Worker, and what lets the
    // nested callback loop install one too. Null when there is none.
    // What the process reached (an image's unit, a native module): the
    // page's start-up status (fetch-note.mjs).
    noteStatus(status) {
      // The process's own .exe, once its main image is in: said with each unit
      // fetch, so the page can name the program the code is for.
      if (status?.program !== undefined) this.owner = String(status.path ?? '').split(/[\\/]/).pop();
      noteStatus(status);
    }
    owner = '';
    // A unit this Worker published, compiled by the host off the guest's
    // threads and handed to every Worker (host.mjs `prefetchUnit`): from the
    // page's server only, not the page's own preloaded files.
    prefetchUnit(key) {
      const c = msg.cgen;
      if (!c?.url || preClosed || preFiles.has(`${key}/c.wasm`)) return;
      let url;
      try { url = new URL(`${c.url}/${key}/c.wasm${c.zst ? '.zst' : ''}`, globalThis.location?.href).href; } catch { return; }
      port.post({ kind: 'prefetchUnit', key, url });
    }
    loadUnit(key) {
      // The entries now; the module's bytes only if this Worker compiles it
      // (most get the Module from a sibling: see `Gex#cgenModule`). Fetching
      // every unit's bytes in every thread's Worker, only to use a shared
      // Module, cost each new guest thread seconds at start-up.
      const entries = this.loadFile(`${key}/c.entries`);
      if (!entries) return null;
      const env = this;
      let wasm;
      return {
        entries,
        get wasm() {
          wasm ??= env.loadFile(`${key}/c.wasm`);
          return wasm;
        },
      };
    }
    // `len` bytes at `off` of a unit-cache file (a split unit's
    // `cold.bin`), synchronously; null when they cannot be read.
    loadRange(rel, off, len) {
      const c = msg.cgen;
      if (!c) return null;
      try {
        if (c.root) {
          const fs = globalThis.process?.getBuiltinModule?.('node:fs');
          if (!fs) return null;
          const fd = fs.openSync(`${c.root}/${rel}`, 'r');
          try {
            const b = new Uint8Array(len);
            return fs.readSync(fd, b, 0, len, off) === len ? b : null;
          } finally {
            fs.closeSync(fd);
          }
        }
        const x = new XMLHttpRequest();
        x.open('GET', `${c.url}/${rel}`, false);
        x.responseType = 'arraybuffer';
        x.setRequestHeader('Range', `bytes=${off}-${off + len - 1}`);
        // Said to the page, as a unit's own files are: a cold cluster is
        // code the program is waiting for.
        const done = noteFetch(`unit:${rel}`, len, this.owner);
        try {
          x.send();
        } finally {
          done(x.response?.byteLength ?? 0);
        }
        const got = new Uint8Array(x.response ?? new ArrayBuffer(0));
        // 206 is the range; a server without ranges sends the whole file.
        const at = x.status === 206 ? 0 : x.status === 200 ? off : -1;
        return at >= 0 && got.length >= at + len ? got.slice(at, at + len) : null;
      } catch {
        return null;
      }
    }
    // The zstd decoder made above, or null.
    get zstd() {
      return zstd;
    }
    // Any file of the unit cache by its relative path (an export
    // provider's module: `providers/<name>-<fingerprint>.wasm`), or null.
    loadFile(rel) {
      const c = msg.cgen;
      if (!c) return null;
      // Fetched by the page ahead of time (an ejected page's preload list).
      const pre = preFiles.get(rel);
      if (pre) return pre.slice(); // an unshared copy: Module and TextDecoder refuse shared views
      // A late unit's (eject-site.mjs --late), in the fill: a Worker started
      // before the page had compiled it waits for its bytes there.
      const late = fillFile(`unit:${rel}`);
      if (late) return late;
      if (preClosed) return null;
      if (unitFiles.has(rel)) {
        const b = unitFiles.get(rel);
        return b ? b.slice() : null;
      }
      try {
        if (c.root) {
          const fs = globalThis.process?.getBuiltinModule?.('node:fs');
          return fs ? new Uint8Array(fs.readFileSync(`${c.root}/${rel}`)) : null;
        }
        // A file the runtime package lists by its own URL (a bundler names
        // them where it puts them): not one it lists is not there.
        if (c.files && !c.files[rel]) return null;
        // A unit's module, served compressed (`c.zst`): decoded here.
        const packed = c.zst && zstd && rel.endsWith('/c.wasm');
        const x = new XMLHttpRequest();
        x.open('GET', c.files ? c.files[rel] : `${c.url}/${rel}${packed ? '.zst' : ''}`, false);
        x.responseType = 'arraybuffer';
        const done = noteFetch(`unit:${rel}`, 0, this.owner);
        const tx = logNamed('unitstall') ? performance.now() : -1;
        try {
          x.send();
        } finally {
          done(x.response?.byteLength ?? 0);
        }
        if (tx >= 0) console.warn(`[unitxhr] ${workerGex?.who?.pid ?? '?'}: ${rel} ${x.status} ${(performance.now() - tx).toFixed(1)} ms`);
        // A unit with no compressed copy: its plain one.
        if (packed && x.status === 404) {
          x.open('GET', `${c.url}/${rel}`, false);
          x.send();
        }
        const unpack = packed && x.responseURL?.endsWith('.zst');
        const got = x.status !== 200 ? null : unpack ? zstd.decode(new Uint8Array(x.response)) : new Uint8Array(x.response);
        // Small files, and the absent, to the page for the Workers to come
        // (a forked child places every unit its parent had): not the
        // module's bytes, which the page holds compiled.
        if ((got === null ? x.status === 404 : got.byteLength <= UNIT_FILE_MAX) && !rel.endsWith('/c.wasm') && typeof SharedArrayBuffer === 'function') {
          let bytes = null;
          if (got) {
            bytes = new Uint8Array(new SharedArrayBuffer(got.byteLength));
            bytes.set(got);
          }
          unitFiles.set(rel, bytes);
          port.post({ kind: 'unitFile', rel, bytes });
        }
        return got;
      } catch {
        return null;
      }
    }
  }

  if (msg.debugExit) {
    globalThis.GEX_DEBUG_EXIT = 1;
    // Keep the engine's wasm-frame stack for a trap, before the shim rewraps it
    // into a plain Error: the wasm frames say *where* a syscall trap happened.
    globalThis.GEX_TRAP_STACK = 1;
  }
  // A 64-bit guest memory: the wasm64 shim's process.
  if (!(SP_CODE && single.on)) {
    if (msg.mem64) globalThis.GEX_MEM64 = 1;
    else delete globalThis.GEX_MEM64;
  }
  // Which flavours the page runs, for an exec's choice (vfork.rs `exec_target`).
  if (msg.pageFlavours !== undefined) globalThis.GEX_PAGE_FLAVOURS = msg.pageFlavours;
  otherGate(msg);
  if (msg.wasmDlls) Gex.wasmDlls = msg.wasmDlls;
  if (msg.elfUnits) Gex.elfUnits = msg.elfUnits;
  if (msg.providerSkip) Gex.providerSkip = msg.providerSkip;
  if (msg.providerSources) Gex.providerSources = msg.providerSources;
  if (msg.stateful) Gex.stateful = msg.stateful;
  if (msg.fjitLibs) Gex.fjitLibs = msg.fjitLibs;
  if (msg.identities) Gex.identities = msg.identities;
  if (msg.testBreakStackGuard) globalThis.GEX_TEST_BREAK_STACK_GUARD = 1;
  if (msg.testUd2Entry) globalThis.GEX_TEST_UD2_ENTRY = 1;
  if (msg.fillTime) globalThis.GEX_FILL_TIME = 1;
  if (msg.unitCache) globalThis.GEX_UNIT_CACHE = 1;
  // Lazy files the page is filling in the background (app.js `fill`): their
  // reads wait for the bytes there, not a request of their own (lazy-fetch.mjs).
  if (msg.fill) setFill(msg.fill);
  // The page's cache of lazy files (app.js `lazyCache`, lazy-cache.mjs).
  if (msg.lazyCache) setLazyCache(msg.lazyCache);
  // The directories the page serves (page/runtime fs.mount, page-fs.mjs).
  if (msg.pageFs) setPageFs(msg.pageFs);
  // A rest bundle's files among the lazy ones (lazy-fetch.mjs `lazyReady`).
  // Its decoder now: a runtime the pool made ahead for an exec'd child
  // started its kernel before this message said so, and found none needed.
  if (msg.restBundle) {
    globalThis.GEX_REST_BUNDLE = 1;
    await lazyReady();
  }
  // `kernelStack` is this Worker's slice of the kernel's stack array. Absent for
  // the first Worker, which keeps the linked-in stack and reads the numbers out.
  // The runtime made in the pool, when this process runs on a fresh memory
  // of its own as that one does, on the same kernel stack.
  const pre = preattached && msg.kind === 'adopt' && msg.child?.exec && !msg.memory
    && !msg.recycledMemory && !msg.moduleStore ? await preattached.catch(() => null) : null;
  // Made for an exec'd child and taken by something else: its memory is
  // dropped unused, and stays committed until this Worker's isolate collects
  // (host.mjs `#preattach`; a test counts these).
  if (preattached && !pre) vforkTrace(`[pool] pid ${msg.child?.pid ?? msg.pid ?? '?'}: a runtime made ahead was left unused (${msg.kind})`);
  preattached = null;
  // A runtime made in the pool that does not fit the process it was made
  // for is a host bug, not a choice: said, and the process gets a new one.
  const fits = pre && pre.shared === !!msg.shared && pre.stack === msg.kernelStack
    && pre.mem64 === !!msg.mem64;
  if (pre && !fits) {
    console.error(`[gex] pid ${msg.child?.pid}: the pooled runtime does not fit its process `
      + `(shared ${pre.shared}/${!!msg.shared}, stack ${pre.stack}/${msg.kernelStack}, mem64 ${pre.mem64}/${!!msg.mem64}); a new one is made`);
  }
  // Back to the process's own type, should the pool's have been the other.
  if (pre && pre.mem64 && !msg.mem64 && !(SP_CODE && single.on)) delete globalThis.GEX_MEM64;
  let gex;
  try {
  gex = fits
    ? pre.gex
    : new Gex(new WorkerEnv(), {
      kernel: msg.kernel ? { ...msg.kernel, stackTop: msg.kernelStack } : null,
      // A thread joins its process's memory; a process makes one. `shared` only
      // decides whether the memory a process builds can have threads later.
      // A fork child may be given an exited process's memory, zeroed
      // (`recycledMemory`, host.mjs): the same as a new one, its pages touched.
      memory: msg.memory ?? msg.recycledMemory ?? null,
      moduleStore: msg.moduleStore ?? null,
      shared: !!msg.shared,
      modules: pooled,
      fiber: rawPort.fiber ?? null,
      mem64: !!msg.mem64,
    });
  } catch (e) {
    // No memory for the process (`wasmMemory`: the tab holds as many as it
    // can): it ends as a crash, said plainly (handle's caller), and the
    // kernel hears of it here, since there is no runtime to say it.
    if (!msg.memory) await endUnstarted(msg).catch((err) => console.error(`[gex] pid ${msg.child?.pid ?? msg.pid ?? '?'}: its end was not recorded: ${err?.message ?? err}`));
    throw e;
  }
  if (gex === pre?.gex) gex.setEnv(new WorkerEnv());
  // Module-scope, so a `sharedModule` message (which arrives outside this
  // handler's run/adopt/thread path) can reach this Worker's runtime.
  workerGex = gex;
  gex.setDoorWord(msg.doorWord);
  if (msg.door) gex.setDoor(msg.door);
  if (msg.cgen?.mode) gex.useCgen(msg.cgen.mode);
  if ((msg.providers ?? msg.opts?.providers) === 0) gex.noElfProviders();
  // Any sibling modules the host already had when this Worker started.
  for (const [key, mod] of msg.sharedModules ?? []) gex.addSharedModule(key, mod);
  for (const [key, mod] of msg.preModules ?? []) if (key.startsWith('cgen:') || key.startsWith('cgenfile:')) gex.addSharedModule(key, mod);
  if (globalThis.GEX_SHARE_TRACE || globalThis.process?.env?.GEX_SHARE_TRACE || msg.shareTrace) console.warn(`[share] ${msg.kind} start with ${(msg.sharedModules ?? []).length} shared modules (${(msg.sharedModules ?? []).filter(([k]) => String(k).startsWith('cgen:')).length} units, ${(msg.sharedModules ?? []).filter(([k]) => String(k).startsWith('wasm:')).length} programs)`);

  // The GL device (gles-gears): the page's GlDevice, reached over a ring in a
  // SharedArrayBuffer. `useGl` sets the flag `gex_set_gl` reads at load, so it
  // must run before `load` -- which mounts /dev/gex-gl only when the flag is on.
  if (msg.glChannel?.inRealm) {
    // The single-thread build: the compositor's Mesa is in this realm
    // (page/src/compositor-worker.js `glInRealm`), and a submit is a promise
    // the Suspending `gl_submit` waits on. A batch is run straight out of
    // this process's memory: no copy, and no batch in flight unwaited.
    // An unwaited batch is waited for all the same; its answer is what the
    // Worker build's would be, the frames shown its PRESENT replied.
    const { shownInReply } = await bridge('gl');
    gex.useGl({
      submit: (memory, ring, client, unwaited) => {
        if (!globalThis.__gexGlInRealm) return -1;
        const r = globalThis.__gexGlInRealm.submit(memory, ring, client);
        if (!unwaited) return r;
        return Promise.resolve(r).then((rc) => (rc < 0 ? rc : shownInReply(new Uint8Array(memory.buffer), ring)));
      },
    });
  } else if (msg.glChannel) {
    const { GlProxy } = await bridge('gl');
    // Posted only to a device that does not wait on the channel itself
    // (GlProxy): an OffscreenCanvas renderer waits on the control word, and
    // thousands of synchronous queries a second stay off the event loops.
    gex.useGl(new GlProxy(msg.glChannel, () => port.post({ kind: 'gl' })));
  }
  // The GL device runs D3D too: /dev/gex-d3d beside /dev/gex-gl.
  if (msg.glChannel?.d3d) gex.useD3d(true);
  // The Wayland compositor: this process's window into the scene arena the
  // compositor worker draws from. `useWl` sets the flag `gex_set_wl` reads at
  // load, so it must run before `load` -- which binds `wayland-0` only when it
  // is on.
  if (msg.wlChannel) {
    const { WlProxy, outputSize } = await bridge('wl');
    // The token is this process's pid, which is not known yet -- the process
    // has none until it is loaded or adopted -- so it is asked for lazily, on
    // the first publish.
    const proxy = new WlProxy(msg.wlChannel, () => port.post({ kind: 'wl' }), () => gex.wlToken());
    gex.useWl(proxy);
    // The output size a toplevel is configured to, before the guest runs.
    const out = outputSize(msg.wlChannel);
    gex.wlOutput(out.w, out.h, out.scale);
    // Sent when the token is known, which is after load: `onLoaded` is the
    // first moment this process has a pid. With pixel buffers in the channel
    // (wl-bridge.mjs) the compositor never needs this process's memory --
    // and must not hold it -- so only the global kernel's goes, once a page,
    // for the wake. Without them (the single-thread build) the compositor
    // reads pixels out of the memory: the object, not its buffer, since a
    // shared memory that grows gives out a new buffer.
    gex.onWlToken = (token) => {
      recyclable = false;
      const kernel = proxy.firstForKernel() ? gex.kernelWake()?.memory : undefined;
      const pixels = !!msg.wlChannel.pixels;
      if (pixels && !kernel) return;
      port.post({ kind: 'wl-memory', token, memory: pixels ? undefined : gex.memory, kernel });
    };
    proxy.onToken = gex.onWlToken;
    proxy.onPixels = (p) => port.post({ kind: 'wl-pixels', ...p });
    proxy.wakeAddr = () => gex.kernelWake();
  } else if (msg.wlStub) {
    // A headless compositor for node: the shim's gex-wayland runs and binds
    // `wayland-0`, so a guest connects and inits exactly as in the browser, but
    // there is no canvas and no input. What the browser's compositor worker
    // does on WebGPU, this does in software -- painter's algorithm over the
    // published scene -- which is enough to prove a client rendered.
    //
    // GEX_WL_DUMP=<path>: composite the scene to a PPM so a render can be
    // eyeballed. GEX_COVERAGE_OUT=<path>: dump every block PC translated in
    // this process (the reached set) as hex lines, to seed a
    // reachability-trimmed closure build. Re-dumped as the run reaches more
    // code; the set only grows, so the last write is the fullest.
    const dumpPath = globalThis.process?.env?.GEX_WL_DUMP;
    const covPath = globalThis.process?.env?.GEX_COVERAGE_OUT;
    const fs = (dumpPath || covPath) ? await import('node:fs') : null;
    const wlb = await bridge('wl');
    const dumpCoverage = () => {
      if (!fs || !covPath) return;
      try {
        const cov = gex.coverage();
        if (!cov) return;
        let out = '';
        for (let i = 0; i < cov.length; i++) out += '0x' + cov[i].toString(16) + '\n';
        fs.writeFileSync(covPath, out);
        console.error(`[cov] wrote ${cov.length} block PCs to ${covPath}`);
      } catch (e) { console.error(`[cov] dump failed: ${e.message}`); }
    };
    let count = 0;
    const outputPlan = (globalThis.process?.env?.GEX_WL_STUB_OUTPUT ?? '').split(',').filter(Boolean).map((e) => {
      const [, w, h, at] = /^(\d+)x(\d+)(?:@(\d+))?$/.exec(e.trim()) ?? [];
      return { size: ((w & 0xffff) << 16 | (h & 0xffff)) >>> 0, at: +(at ?? 0) };
    });
    let outputStart = null;
    let outputSeen = 0;
    const outputNow = () => {
      if (!outputPlan.length) return 0;
      outputStart ??= perf.now();
      const t = perf.now() - outputStart;
      let size = 0;
      for (const p of outputPlan) if (p.at <= t) size = p.size;
      return size;
    };
    const stubLog = !!globalThis.process?.env?.GEX_WL_STUB_LOG;
    const stubSum = !!globalThis.process?.env?.GEX_WL_STUB_SUM;
    gex.useWl({
      publish(memory, ptr, n) {
        count++;
        if (count === 1 || count % 30 === 0) dumpCoverage();
        const recs = [];
        const words = new Int32Array(memory.buffer, ptr >>> 0, n * wlb.REC);
        for (let i = 0; i < n; i++) {
          const r = i * wlb.REC;
          recs.push({
            surface: words[r + wlb.R_SURFACE],
            role: words[r + wlb.R_ROLE],
            x: words[r + wlb.R_REL_X],
            y: words[r + wlb.R_REL_Y],
            w: words[r + wlb.R_WIDTH],
            h: words[r + wlb.R_HEIGHT],
            addr: wlb.recordAddr(words, r),
            stride: words[r + wlb.R_STRIDE],
          });
        }
        const windows = recs.filter((s) => s.role === wlb.ROLE_TOPLEVEL && s.w > 0);
        // GEX_WL_STUB_LOG: every publish, on the guest's clock (hostNowNs's
        // base), for a test.
        if (stubLog) {
          console.error(`[wlstub] publish t=${(performance.timeOrigin + performance.now()).toFixed(1)} surfaces=${recs.length} windows=${windows.length}`);
        } else if (count === 1 || count % 60 === 0) {
          console.error(
            `[wlstub] publish #${count}: ${recs.length} surfaces, ${windows.length} windows`
            + windows.map((w) => ` ${w.w}x${w.h}`).join(''),
          );
        }
        // GEX_WL_STUB_SUM: each window's pixels as the compositor reads them,
        // summed, for a test (a pool above 4 GiB).
        if (stubSum) {
          const px = new Uint8Array(memory.buffer);
          for (const s of windows) {
            let sum = 0;
            for (let y = 0; y < s.h; y++) {
              for (let x = 0; x < s.w; x++) {
                const i = s.addr + y * s.stride + x * 4;
                sum = (sum + (px[i] | px[i + 1] << 8 | px[i + 2] << 16 | px[i + 3] << 24)) >>> 0;
              }
            }
            console.error(`[wlstub] window ${s.w}x${s.h} at 0x${s.addr.toString(16)} sum=${sum}`);
          }
        }
        if (!fs || !dumpPath || !windows.length) return;
        // The output is the largest window's size, and every surface is drawn
        // into it at its own offset: one file that shows the whole session.
        const W = Math.max(...windows.map((s) => s.x + s.w));
        const H = Math.max(...windows.map((s) => s.y + s.h));
        const body = Buffer.alloc(W * H * 3);
        const px = new Uint8Array(memory.buffer);
        for (const s of recs) {
          if (s.role === wlb.ROLE_CURSOR || s.w <= 0 || s.h <= 0) continue;
          for (let y = 0; y < s.h; y++) {
            const oy = s.y + y;
            if (oy < 0 || oy >= H) continue;
            for (let x = 0; x < s.w; x++) {
              const ox = s.x + x;
              if (ox < 0 || ox >= W) continue;
              const i = s.addr + y * s.stride + x * 4;
              const o = (oy * W + ox) * 3;
              body[o] = px[i + 2];
              body[o + 1] = px[i + 1];
              body[o + 2] = px[i];
            }
          }
        }
        fs.writeFileSync(dumpPath, `P6\n${W} ${H}\n255\n`);
        fs.appendFileSync(dumpPath, body);
        if (count === 1) console.error(`[wlstub] wrote ${W}x${H} scene to ${dumpPath}`);
      },
      pollInput() { return 0; },
      // No input here, ever (`wl_input_pending`, which the shim asks of every
      // compositor since the pump learned to wait for a client's input) --
      // only a resize, as the page's proxy says one (`WlProxy.inputPending`).
      inputPending() { return outputNow() !== outputSeen; },
      // GEX_WL_STUB_OUTPUT=1300x713,1000x650@2000,...: the output's size, and
      // what it becomes that many ms after the pump first asks, for a test of
      // a resize reaching connected clients. Unset: none, as before.
      outputSize() { outputSeen = outputNow(); return outputSeen; },
    });
  }

  // The Node fast lane's network: real TCP sockets rather than Wisp, so the
  // paths above them -- apt's helper, the fetch, the parse -- run here in two
  // minutes instead of twenty.
  if (msg.network) {
    const { NodeNetwork } = await import('./net-node.mjs');
    const net = new NodeNetwork({ pageHosts: msg.pageHosts ?? null, tcp: !!msg.tcp });
    if ((SP_CODE && single.on)) net.onActivity = () => sched.wake();
    gex.useNetwork(net);
  }
  if (msg.pageNet) {
    // The single-thread build: the page's sockets, the object itself (every
    // fiber is in its realm). Their traffic wakes parked fibers.
    msg.pageNet.onActivity = () => sched.wake();
    gex.useNetwork(msg.pageNet, { shared: true });
  } else if (msg.netChannel) {
    // The page's sockets, called across shared memory (net-bridge.mjs): a
    // wasm32 program never hands this Worker's event loop back, so a
    // connection of its own here would never deliver.
    const { NetProxy } = await bridge('net');
    gex.useNetwork(new NetProxy(msg.netChannel, msg.child?.pid ?? msg.pid ?? 1), { shared: true });
  } else if (msg.pageHosts && !msg.network) {
    // Only the page-served hosts, where the page cannot serve its sockets.
    const { Sockets } = await import('../../../page/src/net.js');
    const net = new Sockets(null, { pageHosts: msg.pageHosts, onEvent: (e) => port.post({ kind: 'net', event: e }) });
    gex.useNetwork(net);
  }
  // A Worker given back after an earlier process (a test
  // counts these).
  if (runs > 1 && msg.kind !== 'thread') {
    vforkTrace(`[pool] pid ${msg.child?.pid ?? msg.pid ?? '?'}: given-back worker, its process ${runs}`);
  }
  try {
    let entry;
    // A process enters on its first thread (where `run` starts); a thread has
    // registers of its own.
    let ctx;
    // A thread the process lost before this Worker could adopt it.
    let gone = false;
    // A thread's: how many `execve`s its process had made when it came up.
    let execGen = 0;
    if (msg.kind === 'thread') {
      const _t = (s) => { if (globalThis.GEX_ATTACH_TRACE) console.error(`[thr ${msg.tid}] ${s}`); };
      _t(`adopt start pc=0x${(msg.pc??0n).toString(16)} ctx=0x${(msg.ctx??0n).toString(16)}`);
      if (msg.spare) {
        msg.spareGo = () => (spareGot ? Promise.resolve(spareGot.thread ?? null) : new Promise((r) => { spareWaiter = r; }));
      }
      ({ entry, ctx, gone } = await (msg.snap ? gex.snapThread(msg.shim, msg.guest, msg) : gex.adoptThread(msg.shim, msg.guest, msg)));
      if (msg.snap) {
        // The process's other threads came back in Workers of their own
        // (`ProcessHost.restore`), not through this one's `spawnThread`:
        // its memory is theirs too, so its exit does not zero it under them.
        if (msg.snapThreads > 1) recyclable = false;
        port.post({ kind: 'snap-attached' });
        gex.snapRelease();
      }
      execGen = gex.adoptGen ?? gex.shim.exports.gex_exec_gen?.() ?? 0;
      _t(`adopt done entry=0x${(entry??0n).toString(16)}, entering run`);
    } else if (msg.kind === 'adopt') {
      narrate(`[worker] pid ${msg.child?.pid}: adopt arrived`);
      // With the closure, as a fresh process gets it: a child inherits its
      // parent's code through the fork record but no record that those pages
      // are a closure's, so without this an exec'd `apt-get` throws every
      // covered library away and re-translates millions of blocks.
      // The identity too: a forked child never calls `load` (where
      // `gex_set_identity` is said), so without this it runs as uid 1000 and
      // `dpkg` refuses.
      if (msg.child) {
        msg.child.uid = msg.uid ?? msg.opts?.uid;
        msg.child.gid = msg.gid ?? msg.opts?.gid;
      }
      entry = await gex.adoptProcess(msg.shim, msg.guest, msg.child, {
        ...msg,
        closure: msg.closure ?? (await closureModules(msg.closureCount ?? 0)),
        wantLibrary,
      });
      // On the forking thread's registers, which the child's kernel keeps
      // where the parent had them (`gex_adopt_child`): `run`'s default is the
      // first thread's. A fork from any other thread entered with the first
      // thread's registers -- for 32-bit Wine, Wine's spawn from services.exe's RPC
      // thread started the child's `__gex_thread_start` on that thread's
      // stack, read a stack word as `clone_start`'s table index and trapped
      // ("table index is out of bounds", rpcss and every service child).
      if (msg.child?.ctx) ctx = Number(msg.child.ctx);
    } else {
      entry = await gex.load(
        msg.shim,
        msg.guest,
        msg.elf,
        msg.files ?? {},
        msg.args ?? [],
        // `closure` arrives at the top level (it is in the host's `extra` so a
        // forked child gets it too); everything else about a run is in `opts`.
        {
          ...(msg.opts ?? {}),
          // Whatever arrived in its own message, waited for here.
          closure: msg.closure ?? msg.opts?.closure
            ?? (await closureModules(msg.closureCount ?? 0)),
          // Top level as well as `opts`: a child's Worker gets the host's
          // `extra`, and pid 1's `opts` never reach it.
          link: msg.link ?? msg.opts?.link,
          dense: msg.dense ?? msg.opts?.dense,
          fjit: msg.fjit ?? msg.opts?.fjit,
          fjitMax: msg.fjitMax ?? msg.opts?.fjitMax,
          fjitOnly: msg.fjitOnly ?? msg.opts?.fjitOnly,
          liftMax: msg.liftMax ?? msg.opts?.liftMax,
          inlineDispatch: msg.inlineDispatch ?? msg.opts?.inlineDispatch,
          xlateReport: msg.xlateReport ?? msg.opts?.xlateReport,
          discoverSkip: msg.discoverSkip ?? msg.opts?.discoverSkip,
          t1Fold: msg.t1Fold ?? msg.opts?.t1Fold,
          providers: msg.providers ?? msg.opts?.providers,
          xcache: msg.xcache ?? msg.opts?.xcache,
          wantLibrary,
        },
      );
      if (!entry) throw new Error('the program could not be loaded');
      // The files are in the kernel's filesystem now, copied. The views here
      // are of the page's bundle (one SharedArrayBuffer, ~460 MiB for a
      // demo), which stays resident while anything views it: let go,
      // and tell the page so it can let go of its own.
      if (msg.files) {
        msg.files = null;
        msg.elf = null;
        // A bundle the page moved here rather than shared: freed now, not
        // whenever this Worker's isolate next collects (one that runs a guest
        // in wasm may not for the whole session). Not with
        // `ArrayBuffer.prototype.transfer(0)`: that detaches V8's view, but
        // the browser's own handle on a buffer that arrived by postMessage
        // keeps the bytes until that collection (Chromium: 400 MiB stayed
        // resident either way). Moved again, into a port already closed, the
        // message is dropped and the bytes with it.
        const gone = (msg.transfer ?? []).filter((b) => b instanceof ArrayBuffer && !b.detached);
        if (gone.length && typeof MessageChannel === 'function') {
          const ch = new MessageChannel();
          ch.port2.close();
          ch.port1.postMessage(null, gone);
          ch.port1.close();
        }
        for (const b of gone) if (!b.detached) b.transfer?.(0);
        msg.transfer = null;
        port.post({ kind: 'files-loaded' });
      }
    }
    // Where `/dev/dsp`'s output is, for the page to play (once per page;
    // every process tells it, the first one wins): a loaded process's and an
    // adopted one's, which is every process the runtime API's system starts
    // (host.mjs `spawnExec`).
    if (msg.kind !== 'thread') {
      const ring = gex.audioRing?.();
      if (ring) port.post({ kind: 'audio', memory: ring.memory, at: ring.at });
    }

    // Statically recompile each object once it proves hot, rather than meeting
    // it one dispatch miss at a time. The feature the project exists for, off
    // unless asked for. For a forked child too: its first act is `execve`, and
    // what it maps then has never been seen.
    const recompile = msg.opts?.recompileObjects ?? msg.recompileObjects;
    if (recompile) {
      gex.shim.exports.gex_set_recompile_objects?.(recompile);
    }
    // In `opts` as well: the page passes what it knows about a run through there.
    const beat = msg.heartbeat || msg.opts?.heartbeat;
    if (beat) {
      gex.shim.exports.gex_set_heartbeat?.(BigInt(beat));
    }
    // GEX_PHASES: one call's time by phase (gex_kernel::phase), said every 20,000 stats.
    if (msg.phases || msg.opts?.phases) gex.shim.exports.gex_set_phases?.(msg.phases || msg.opts.phases);
    if (msg.inPlace === 0) gex.shim.exports.gex_set_in_place?.(0);
    if (msg.excTrace) gex.shim.exports.gex_set_exc_trace?.(1);
    if (msg.fcacheStats) gex.shim.exports.gex_set_fcache_stats?.(1);
    if (msg.winsize) {
      gex.shim.exports.gex_set_winsize?.(msg.winsize.rows, msg.winsize.cols);
    }
    // Where the kernel's per-Worker stacks are, so the host can hand out the
    // next slice. Only the first Worker can ask; the rest run on a slice already.
    if (!msg.kernelStack) {
      const stacks = gex.kernelStacks?.();
      if (stacks) port.post({ kind: 'kernelStacks', ...stacks });
    }
    // The host sends a resize to every Worker it has, an exited one too until
    // it hears of the exit: none after this one's slot went back
    // (`Gex#releaseWorker`), whose next holder runs on its stack.
    winsize = (rows, cols) => { if (!gex.released) gex.shim.exports.gex_set_winsize?.(rows, cols); };
    // The line discipline is the guest's choice, read from its termios; the page
    // does the echoing so it has to be told, and it is sent with every write
    // because that is when it changes (a shell sets up then prints a prompt).
    lflag = () => gex.shim.exports.gex_tty_lflag?.() ?? 0;
    const ex = gex.shim.exports;
    // An ejected project's AOT-only run: nothing translates, in any Worker.
    if ((msg.jit ?? msg.opts?.jit) === false) ex.gex_set_jit(0);
    // An ejected project's run, its training run included (translation on).
    if (msg.aot ?? msg.opts?.aot) ex.gex_set_aot?.(1);
    if (msg.hot) ex.gex_set_hot?.(1);
    if (msg.noIntrinsics) ex.gex_set_intrinsics?.(0);
    if (msg.intrinsicMask !== undefined) ex.gex_set_intrinsic_mask?.(msg.intrinsicMask);
    if (msg.trace) ex.gex_set_trace?.(1);
    if (msg.wholePage ?? msg.opts?.wholePage) Gex.wholePage = true;
    if (msg.vforkTrace ?? msg.opts?.vforkTrace) globalThis.GEX_VFORK_TRACE = 1;
    if (msg.traceMisses) ex.gex_set_trace_misses?.(1);
    if (msg.dlChecked ?? msg.opts?.dlChecked) ex.gex_set_dl_checked?.(1);
    if (msg.jitDiag) ex.gex_set_jit_diag?.(1);
    if (msg.traceBlocks) ex.gex_set_trace_blocks?.(1);
    if (msg.watchMem) ex.gex_set_watch_mem?.(BigInt(msg.watchMem));
    if (msg.checkSp) ex.gex_set_check_sp?.(1);
    // Per-process page numbering on (1, the default) or off (0). Off publishes
    // closure dir entries at the build's slot numbers, which the profile-guided
    // wayland closures need (the demand path's streaming interaction corrupts
    // their dispatch); on keeps tables small across a many-fork run like apt.
    if (msg.claimPages !== undefined) ex.gex_set_claim_pages?.(msg.claimPages);
    if (msg.copyLoop) ex.gex_set_copy_loop?.(1);
    if (msg.coherentLoads) ex.gex_set_coherent_loads?.(1);
    // TEST (GEX_DEBUG_PANIC, a test-only switch): the shim panics on purpose.
    if (msg.debugPanic) {
      try {
        ex.gex_debug_panic?.(msg.debugPanic);
      } catch (e) {
        const again = ex.gex_panic_reporting?.()
          ? '; it panicked again inside the panic report, whose message is lost (the first is above)'
          : '';
        console.error(`gex: the shim trapped in gex_debug_panic (${e.message})${again}`);
        throw e;
      }
    }
    if (msg.spGuard) ex.gex_set_sp_guard?.(1);
    if (msg.trapPc) ex.gex_set_trap_pc?.(BigInt(msg.trapPc));
    if (msg.pregrow) ex.gex_set_pregrow?.(1);
    if (msg.wlTrace) ex.gex_set_wl_trace?.(1);
    // A forked or exec'd child never calls `load`, so set the switch here.
    const link = msg.link ?? msg.opts?.link;
    if (link) ex.gex_set_snapshot?.(Number(link));
    if (msg.debugExit) {
      // Before the guest runs an instruction: memory decides how many processes
      // a browser holds, so whether the cost is placing the image or running it.
      port.post({ kind: 'loaded', memoryMib: gex.memory.buffer.byteLength / (1 << 20) });
    }
    // Around `gex.run` alone: everything before it is loading and module
    // compilation, so a codegen A/B measured through the total is a third
    // module size and V8's compiler.
    // A thread exits through syscall 93 without marking the whole process
    // exited, so only a process Worker should label an unreachable return as
    // an abandoned process stack.
    // A restored snapshot's first thread is its process's, as the Worker
    // that ran `load` was (`ProcessHost.restore`).
    const asThread = msg.kind === 'thread' && !msg.snapMain;
    globalThis.GEX_REPORT_ABANDONED = !asThread;
    const runStarted = performance.now();
    // A thread's Worker joins a shared memory that another Worker may have
    // grown since this one instantiated, and the engine updates a Worker's
    // bound on that memory only at its own next interrupt: a thread that
    // starts first reads past the bound it still holds and dies of an
    // out-of-bounds trap (32-bit Wine's second thread, started as its loader
    // reserves its address space, did -- about one process in a hundred
    // under load, which the parent's CreateProcess saw as a child that died
    // at once). Growing by nothing takes the memory's present length.
    if (asThread && !gone && gex.memory) {
      try {
        gex.memory.grow(0);
      } catch {
        // A 64-bit memory counts its pages in BigInt.
        gex.memory.grow(0n);
      }
    }
    let status = gone ? 0 : await gex.run(entry, ctx);
    // A thread that ended while its process runs on as the same program:
    // its Worker waits for one of the process's next threads, as a spare
    // does, if the host keeps it (`park`, answered by `spareGo`), and runs
    // it. A Wine program that starts its threads one after another then
    // waits for no Worker's start between them.
    // Never in the single-thread build: its threads are fibers of one
    // context, and a parked one would keep its process from exiting.
    while (asThread && msg.park && !(SP_CODE && single.on) && !gone && !msg.snap && !ex.gex_thread_gone?.() && !ex.gex_exited?.()
      && (ex.gex_exec_gen?.() ?? 0) === execGen && gex.parkThread?.()) {
      spareGot = null;
      port.post({ kind: 'park', status, gen: execGen });
      const next = { spare: true, parked: true, spareGo: () => (spareGot ? Promise.resolve(spareGot.thread ?? null) : new Promise((r) => { spareWaiter = r; })) };
      ({ entry, ctx, gone } = await gex.adoptThread(null, null, next));
      if (gone) {
        status = 0;
        break;
      }
      msg.tid = next.tid;
      msg.ctx = next.ctx;
      msg.pc = next.pc;
      if (next.procOpts) msg.procOpts = next.procOpts;
      execGen = gex.adoptGen ?? ex.gex_exec_gen?.() ?? 0;
      try {
        gex.memory.grow(0);
      } catch {
        gex.memory.grow(0n);
      }
      status = await gex.run(entry, ctx);
    }
    // The process's exec went to another Worker (a program of the other
    // memory flavour, lib.rs `exec_away`): this one ends, the process does
    // not. The host starts the exec as a posix_spawn child's.
    const away = !gone && !asThread ? gex.execAwayRecord?.() : null;
    if (away) {
      gex.close?.();
      gex.releaseWorker?.();
      port.post({ kind: 'execAway', child: away });
      return;
    }
    // This Worker's thread went with the old program when another thread
    // called `execve`: the process runs on in that thread's Worker, so this
    // one ends as a thread does, never as the process.
    //
    // And the other way round: a thread Worker whose thread made that
    // `execve` (or one since) carries the process now, and ends it as its
    // process Worker would have.
    const carries = asThread && !gone && !ex.gex_thread_gone?.()
      && (ex.gex_exec_gen?.() ?? 0) !== execGen;
    const thread = (asThread && !carries) || !!ex.gex_thread_gone?.();
    // A thread's end, its process going on as the same program: the
    // thread-local blocks its Worker made for `dlopen`ed modules go with it
    // (after an `execve` the addresses are the new program's).
    if (asThread && !gone && !carries && !ex.gex_thread_gone?.() && !ex.gex_exited?.()
      && (ex.gex_exec_gen?.() ?? 0) === execGen) gex.freeThreadTls?.();
    const guestMillis = Math.round(performance.now() - runStarted);
    if (msg.exitTrace && status !== 0) {
      // Who sent the last signal anywhere (the global kernel's record): a
      // signal death names its sender.
      try {
        const k = gex.kernel;
        const v = BigInt.asUintN(64, BigInt(k?.k_last_kill?.() ?? 0));
        console.warn(`[gex] pid ${msg.pid ?? '?'} ended with ${status}; last signal sent: ${Number(v & 0xffn)} from pid ${Number(BigInt.asIntN(24, v >> 40n))} to ${Number(BigInt.asIntN(32, (v >> 8n) & 0xffffffffn))} (#${k?.k_last_kill_seq?.() ?? '?'})`);
      } catch {}
      ex.gex_set_trace?.(1);
      ex.gex_dump_state?.();
      ex.gex_set_trace?.(0);
    }
    // Where each pinned object landed. A pin that does not take runs code built
    // for one address at another; only knowable after `ld.so` maps the libraries.
    if (msg.opts?.pins) ex.gex_pin_report?.();
    // A crashed process still has to die properly: `exit_process` publishes the
    // status and releases descriptors, and a wasm fault abandons the stack
    // without running it -- so a crashed child left its pipe ends shut and no
    // status for its parent, deadlocking `apt-get update`. 134 is `SIGABRT`.
    //
    // Only for a process Worker, never a thread's: `gex_process_gone` sets the
    // process-wide `EXITED` flag, and letting the first thread to exit call it
    // marked the whole process gone -- siblings read `gex_exited()` true, a
    // late-starting thread was stranded `runnable`, and the main thread hung
    // joining it. A thread's own exit already woke its joiner's futex; a thread
    // that faults is handled by the Worker's error path and never reaches here.
    // `?fjit=`: what the function JIT did, and how often its lock was held.
    if (!thread) ex.gex_fjit_report?.();
    if (!thread) ex.gex_process_gone?.(134);
    // `?xcache=`: what this process added to the store, kept for the next load.
    // After the exit is published, so a parent's wait does
    // not wait for it.
    const xcache = msg.xcache ?? msg.opts?.xcache ?? msg.child?.opts?.xcache;
    const xcacheMin = msg.xcacheMin ?? msg.opts?.xcacheMin ?? msg.child?.opts?.xcacheMin;
    if (!thread && xcache) await gex.xcacheSave?.(xcache, xcacheMin ?? 1 << 20, msg.saveGate, msg.xcacheDelay, msg.child?.pid ?? msg.pid ?? 1);
    // `close` before reporting: it drops the tick channel, a live handle that
    // would otherwise keep a Node Worker up.
    gex.close?.();
    // A thread of a program an `execve` replaced: the shim's state is the new
    // program's (its heap and JIT state started again), so nothing more is
    // read from it here.
    const replaced = asThread && !carries && (gone || (ex.gex_exec_gen?.() ?? 0) !== execGen);
    const exitMsg = replaced ? { kind: 'exit', status, guestMillis } : {
      kind: 'exit',
      status,
      threadGone: thread && !asThread ? true : undefined,
      carries: carries || undefined,
      guestMillis,
      exitPc: msg.exitTrace ? Number(ex.gex_current_pc?.() ?? 0n) : undefined,
      // Whatever the runner asked to be read out of the guest before its
      // address space goes away with this thread.
      hot: msg.hot ? hotBlocks(gex) : undefined,
      fmaCalls: msg.fmaCount ? Number(ex.gex_fma_calls?.() ?? -1) : undefined,
      extracted: (msg.extract ?? []).map((p) => [p, gex.readFile(p)]),
      blocks: Number(gex.shim.exports.gex_jit_blocks?.() ?? 0),
      // Of those, ones on a page a closure library claims: work the recompiler
      // already did and this process did again.
      liftInClosure: Number(gex.shim.exports.gex_lift_in_closure?.() ?? 0),
      // Why each miss reached the translator. See `gex_lift_why`: 0 fill not
      // possible yet, 1 no closure owner, 2 owned and already asked, 3 owned/fresh.
      liftWhy: gex.shim.exports.gex_lift_why
        ? [0, 1, 2, 3].map((i) => Number(gex.shim.exports.gex_lift_why(i)))
        : undefined,
      // For each address that translated although its span was filled, whether
      // the slot it dispatches through is written now.
      liftSlots: gex.liftSlots?.(),
      millis: Number(gex.shim.exports.gex_jit_millis?.() ?? 0),
      // What the translator did: instructions lifted, wasm bytes, translations,
      // SMC page invalidations, pages retired, retire calls, -, anonymous blocks.
      // The discovery log (GEX_DISCOVER_SKIP bit 8, run-node's GEX_DL_FILE).
      dl: (() => {
        const e = gex.shim.exports;
        const n = e.gex_dl_text?.(0, 0) >>> 0;
        if (!n) return undefined;
        const p = e.gex_alloc(n) >>> 0;
        e.gex_dl_text(p, n);
        const t = new TextDecoder().decode(new Uint8Array(gex.memory.buffer, p, n).slice());
        e.gex_free?.(p, n);
        return t;
      })(),
      xlate: gex.shim.exports.gex_xlate_stat
        ? [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((i) => Number(gex.shim.exports.gex_xlate_stat(i)))
        : undefined,
      // What the engine cost, which `millis` does not: compiling and
      // instantiating every module this process installed.
      installMs: gex.installMillis,
      // Split: compiling is per bytes and V8 can skip seen bytes, instantiating
      // is per module and paid every time.
      compileMs: gex.compileMillis,
      unitHits: gex.unitHits,
      compiles: gex.compileCount,
      instantiateMs: gex.instantiateMillis,
      growMs: gex.growMillis,
      byObject: gex.blocksByObject,
      cmdline: gex.cmdline,
      // How this process waited, by kind. See `Gex.waits`.
      waits: gex.waits,
      traps: gex.traps,
      loader: gex.loader,
      // What a Worker pays before the guest runs: shim and guest module are
      // instantiated per Worker.
      shimCompileMs: gex.shimCompileMillis,
      shimInstMs: gex.shimInstantiateMillis,
      guestInstMs: gex.guestInstantiateMillis,
      installs: gex.installCount,
      fill: gex.fillStats,
      // How the shared-slot lookup went, to say why a child re-lifts its
      // parent's libraries: pages it already had, pages the identity map knew,
      // pages it did not, pages with no identity.
      reuse: gex.shim.exports.gex_reuse_stat
        ? [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14].map((i) => Number(gex.shim.exports.gex_reuse_stat(i)))
        : null,
      // How much of this process's memory the shim used: a memory per process
      // decides how many processes a browser holds at once.
      heapKib: Number(gex.shim.exports.gex_heap_used_kib?.() ?? -1),
      // What this process's memory grew to -- the question is whether a forked
      // child pays the same again.
      memoryMib: Math.round(gex.memory.buffer.byteLength / (1 << 20)),
      // Dispatch slots the page has left, in thousands. A budget nothing
      // refills: when it runs out, the next module names a table that does not
      // exist and the engine refuses it. Per process, because the question is
      // always how many more processes a run can afford.
      slotsLeftK: Number(gex.shim.exports.gex_slots_left_k?.() ?? -1),
      // Where this process's address space went: guest mappings, the runtime's
      // own heap, Worker stacks. In MiB.
      // What this process's fork copied into it, and what `?forkskip` left out.
      forkCopied: gex.forkCopiedBytes,
      forkSkipped: gex.forkSkippedBytes,
      forkZeros: gex.forkZeroBytes,
      // A posix_spawn child that ran in its parent's Worker until this exec.
      vforkMs: msg.child?.vforkMs,
      arena: gex.shim.exports.gex_arena_spent_mib
        ? [0, 1, 2].map((i) => Number(gex.shim.exports.gex_arena_spent_mib(i)))
        : null,
    };
    // Last: nothing calls into this instance after (`releaseWorker`).
    gex.releaseWorker?.();
    // This process's memory, zeroed, for the host to give the next fork child
    // (`Gex#recycleMemory`): its pages are touched, which a fresh memory's are
    // not. After the kernel has the process gone, so no parent waits on it.
    if (recyclable && !thread && !msg.child?.exec) {
      exitMsg.recycle = gex.recycleMemory?.() ?? undefined;
    }
    // Whether this Worker can go back to the pool for another process
    // (host.mjs `#giveBack`): only a process that exited by `exit` (a status
    // under 128; a signal death is 128 and up), in this Worker alone (no
    // thread, nothing else holding its memory), on no connection of this
    // realm's own (Node's sockets, a Wisp WebSocket). The single-thread
    // build's fibers are never given back.
    exitMsg.workerRuns = runs;
    exitMsg.giveBack = !(SP_CODE && single.on) && !thread && !carries && !gone && recyclable
      && (msg.kind === 'adopt' || msg.kind === 'run')
      && status >= 0 && status < 128 && !msg.network && !msg.pageHosts;
    port.post(exitMsg);
  } catch (e) {
    // A thread that failed before its claim gives its promised slot back.
    gex?.keepPromise?.();
    // An instance that got no slot has no thread-locals: any call into it
    // writes over the thread contexts at address zero (`Gex#noSlot`).
    const inert = !!gex?.noSlot;
    // Where the guest was, not only what the engine said: a crash inside the
    // shim reaches the host as `unreachable`, and the guest pc and object are
    // readable here and nowhere else.
    let where = '';
    try {
      const ex = inert ? null : gex.shim?.exports;
      // Only if this Worker got as far as having a kernel: a failure during
      // load left no host, and `gex_current_pc` panicking on the way in
      // replaced the real error with a bare `unreachable`.
      if (ex?.gex_current_pc && ex.gex_has_host?.()) {
        const pc = ex.gex_current_pc();
        where = ` at guest pc=0x${(pc < 0n ? pc + (1n << 64n) : pc).toString(16)}`;
        if (ex.gex_object_at) {
          const at = ex.gex_alloc(160) >>> 0;
          const n = ex.gex_object_at(pc, at, 160) >>> 0;
          if (n) {
            where += ` in ${utf8(
              new Uint8Array(gex.memory.buffer, at, n),
            )}`;
          }
        }
      }
    } catch {
      // A shim too broken to ask. The engine's message is all there is.
    }
    // The global kernel has no console of its own, so a panic there unwinds to
    // a bare `unreachable`. Its message is in its own memory, readable here
    // because this Worker made the instance.
    try {
      const k = gex.kernel;
      if (k?.k_panic_len?.()) {
        const n = k.k_panic_len() >>> 0;
        const at = k.k_panic_ptr() >>> 0;
        where += ` -- the global kernel panicked: ${utf8(
          new Uint8Array(msg.kernel.memory.buffer, at, n),
        )}`;
      }
    } catch {
      // Nothing to add.
    }
    // Always, not under a flag: the evidence above was printed only under
    // `GEX_DEBUG_EXIT` and thrown away otherwise. A debug switch should add
    // detail to a failure, never be the condition for seeing one.
    console.error(`gex: worker threw: ${e?.message ?? e}${where}\n${e?.stack ?? ''}`);
    // Before the message: until the table knows this process is gone, its
    // parent's `wait4` is parked on a child that will never report. See
    // `gex_abandon` -- #69.
    try {
      // Dump guest registers first: a trap gives JavaScript no address.
      // Not for an instance with no slot: its thread never ran, and the
      // process runs on in its other Workers.
      if (!inert) {
        gex?.shim?.exports?.gex_dump_state?.();
        const abandon = gex?.shim?.exports?.gex_abandon;
        // In the single-thread build the kernel's calls may suspend, which
        // needs a stack of its own (`WebAssembly.promising`).
        if (abandon) await ((SP_CODE && single.on) ? WebAssembly.promising(abandon)(-9) : abandon(-9));
      }
    } catch {
      // The shim is what just fell over; there may be nothing left to call.
    }
    // A fiber's timers outlive it unless stopped (single.mjs).
    if ((SP_CODE && single.on)) gex?.close?.();
    port.post({
      kind: 'threw',
      message: (e?.message ?? String(e)) + where,
      stack: e?.stack ?? '',
    });
    // The trap abandoned this instance's stack; its slot is free for the next.
    gex?.releaseWorker?.(true);
  }
}

// A payload handed over at construction, if the host had one. The handshake it
// replaces -- post `ready`, be sent `adopt` -- costs one turn of the starting
// thread's event loop, and under Node that thread may be a guest about to block
// (`bash -c 'echo $(echo hi)'`), so `ready` never gets delivered and the child
// never starts. `workerData` removes the dependency. `ready` is still posted
// because a browser host has no `workerData`; sending both is harmless.
const handed = await (async () => {
  if (typeof self !== 'undefined') return null;
  try {
    const { workerData } = await import('node:worker_threads');
    return workerData?.gexStart ?? null;
  } catch {
    return null;
  }
})();

port.post({ kind: 'ready' });

// The bridges every process of a page sets up before it runs (GL, Wayland,
// network), loaded now, while this Worker waits in the host's pool: imported
// only when a process arrived, they were about a quarter of a forked child's
// start (12.8, 7.3 and 5.6 ms of 104 on apt's children, timed per phase). A
// later `import` of each is then the module already evaluated.
if (typeof self !== 'undefined') {
  for (const k of Object.keys(bridgeImports)) bridge(k).catch(() => {});
}
if (handed && !carry) handle(handed);
}

// A real Worker runs itself; the single-thread build registers the function
// and runs it once per fiber (single.mjs `FiberWorker`).
// Not a top-level await, which a bundler's classic-script Worker cannot
// hold; a failure is thrown again outside the promise, so the Worker's
// `error` event still reports it.
if ((SP_CODE && single.on)) registerWorkerMain(workerMain);
else channel().then(workerMain).catch((e) => setTimeout(() => { throw e; }));
