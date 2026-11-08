// A Worker's own instance of the global kernel, for the page's helpers that
// write or read the guest's filesystem while it runs (drop-worker.js). The helper calls `k_fs_*` with its
// paths and bytes in a scratch "process" memory of its own; a Worker and not
// the page because the kernel's locks may wait, and the page's thread cannot.

import { kernelThread, lazyFetch, processMemories, wasmMemory, wasmMemoryGone } from '../../hosts/browser/js/lazy-fetch.mjs';
import { blockIo } from '../../hosts/browser/js/blocks.mjs';
import { pageFs, pageFsSuspending, setPageFs } from '../../hosts/browser/js/page-fs.mjs';

/**
 * Instantiate the kernel `spec` ({module, memory}: runtime.js's
 * `globalKernel()`) on the kernel stack whose top is `stackTop` (a
 * `host.takeKernelStack` slice, or a function of the exports giving one, such
 * as a `helperStackTop`), with its thread-local block at the top of it.
 * `suspending`: the single-thread build's, whose page directories answer
 * on a stack that suspends (each call made through `WebAssembly.promising`).
 * Returns {k: the exports, mem: the scratch memory}.
 */
export async function helperKernel(spec, stackTop, pages = 16, { suspending = false } = {}) {
  // The process memory as the kernel build declares it: shared (the page's
  // build) or not; a memory import's sharedness is part of its type.
  // The page's mounts (page/runtime fs.mount), when it serves any.
  if (spec.pageFs) setPageFs(spec.pageFs);
  let inst;
  let mem;
  for (const shared of [true, false]) {
    mem = wasmMemory({ initial: pages, maximum: 65535, shared }, "a helper's scratch memory", spec.spare);
    try {
      inst = await WebAssembly.instantiate(spec.module, {
        env: { memory: spec.memory, ...processMemories(mem, false, spec.spare), lazy_fetch: lazyFetch(spec.memory), block_io: blockIo(spec.memory), page_fs: suspending ? pageFsSuspending(spec.memory) : pageFs(spec.memory) },
      });
      break;
    } catch (e) {
      wasmMemoryGone("a helper's scratch memory", spec.spare);
      if (!shared) throw e;
    }
  }
  const k = inst.exports;
  const top = typeof stackTop === 'function' ? stackTop(k) : stackTop;
  if (top && k.__stack_pointer) kernelThread(k, top);
  return { k, mem };
}
