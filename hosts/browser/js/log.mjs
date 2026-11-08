// What gex says on the console by default, and the switch for the rest.
//
// By default the page runtime says only: a process starting (and the file it
// runs), a library loaded for the first time in this session, and what went
// wrong. Everything else -- Worker hand-offs, fork costs, per-process exit
// numbers, unit hits, the shim's narration -- is debug output behind
// `GEX_LOG`: `?log=1` on the page (or `extra.log`, which every Worker is
// handed), the GEX_LOG environment variable under Node, as natively.
//
//   GEX_LOG=1 (or all)   everything, as before
//   GEX_LOG=debug        the narration (Worker hand-offs, fork costs, the shim's lines)
//   GEX_LOG=record       the `[gex-record]` lines (site/record-wine32.mjs)
//   GEX_LOG=unit         a line per function-tier unit placed in a process
//   GEX_LOG=unitstall    timed lines for unit installs: each first call, catch-up,
//                        publish, sync fetch, and the page's compiles ahead
//                        (only when named)
//   GEX_LOG=wasmlib      a line per native wasm32 module a process installs: its
//                        compile (or that the Worker had it) and instantiation
//                        (only when named)
//   GEX_LOG=spare        spare thread Workers: a program's offer, each spare
//                        started, caught up and given its thread (only when named)
//   GEX_LOG=win          a line when a toplevel appears, first shows pixels and
//                        is retitled, in the page's desktop (only when named)
//   GEX_LOG=dlpack       a line per wasm32 library whose exports the page packs
//                        (gex.mjs `packExports`): its size and time (only when named)
//   GEX_LOG=nodlpack     no packing: every export kept, as before (only when named)
//   GEX_LOG=bridges      the unit entries each Worker entered through a bridge
//                        (only when named: it puts a call on every such entry)
//   GEX_LOG=traps        per process at its end, how its Worker's traps went on:
//                        straight on (`gex_after_trap`) or through each step
//                        (only when named)
//   GEX_LOG=mem          a line per wasm memory the tab makes or lets go, with
//                        how many it holds (lazy-fetch.mjs `wasmMemory`; a
//                        tab holds about 124) (only when named)
//   GEX_LOG=a,b          several of those
//   GEX_LOG=quiet        the default lines but the routine ones: no line per
//                        process started or library loaded; errors and rare
//                        events still said (the runtime API's default)
//   GEX_LOG=off          not even the default lines (tests that compare a
//                        program's output with its native run)

// The page's own `?log=` (a Worker's `location` is its script's, so Workers
// are handed it in their start message instead).
let fromQuery;
function query() {
  if (fromQuery === undefined) {
    fromQuery = null;
    try {
      if (typeof window !== 'undefined' && window.location?.search) fromQuery = new URLSearchParams(window.location.search).get('log');
    } catch {}
  }
  return fromQuery;
}

function value() {
  const v = globalThis.GEX_LOG ?? globalThis.process?.env?.GEX_LOG ?? query();
  return v === undefined || v === null || v === '' || v === '0' ? null : String(v);
}

/** `GEX_LOG=off` (or `none`): the default lines are off too. */
export function logQuiet() {
  const v = value();
  return v === 'off' || v === 'none';
}

/** Whether the routine default lines are said: a process starting, a
 *  library loaded. Off under `quiet` and `off`. */
export function logRoutine() {
  const v = value();
  return v !== 'quiet' && v !== 'off' && v !== 'none';
}

/** Whether debug output of kind `what` (or any, without one) is on. */
export function logOn(what) {
  const v = value();
  if (v === null || v === 'off' || v === 'none' || v === 'quiet') return false;
  if (v === '1' || v === 'all') return true;
  const kinds = v.split(',');
  return what === undefined ? kinds.length > 0 : kinds.includes(what);
}

/** Whether kind `what` is named outright (not just on through `1`/`all`). */
export function logNamed(what) {
  const v = value();
  return v !== null && v.split(',').includes(what);
}

/** Set from a Worker's start message (`extra.log`): a browser Worker sees
 *  neither the page's globals nor its query. */
export function setLog(v) {
  if (v !== undefined && v !== null && globalThis.GEX_LOG === undefined) globalThis.GEX_LOG = String(v);
}
