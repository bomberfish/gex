// fs-opened.js's Worker: one `k_fs_opened` call against the global kernel.
// Message in: { module, memory }; out: { paths } or { error }.
//
// It runs on a kernel stack slot of its own (lazy-fetch.mjs `HELPER_STACKS`):
// it may ask while the page's terminal does. fs-opened.js asks
// one at a time, so two of these never share it.

import { helperStackTop, kernelThread, lazyFetch, processMemories, wasmMemory } from '../../hosts/browser/js/lazy-fetch.mjs';
import { blockIo } from '../../hosts/browser/js/blocks.mjs';
import { pageFs } from '../../hosts/browser/js/page-fs.mjs';

const SCRATCH = 0x1000;

self.onmessage = async (e) => {
  try {
    let inst, mem;
    // The process memory as the kernel build declares it (shared or not).
    for (const shared of [true, false]) {
      mem = wasmMemory({ initial: 1, maximum: 65535, shared }, 'the open-files scratch memory');
      try {
        inst = await WebAssembly.instantiate(e.data.module, { env: { memory: e.data.memory, ...processMemories(mem), lazy_fetch: lazyFetch(e.data.memory), block_io: blockIo(e.data.memory), page_fs: pageFs(e.data.memory) } });
        break;
      } catch (err) {
        if (!shared) throw err;
      }
    }
    const k = inst.exports;
    if (!k.k_fs_opened) throw new Error('this kernel has no k_fs_opened');
    if (k.k_stack_area && k.__stack_pointer) {
      // Its own slot (`HELPER_STACKS`), thread-locals at the top of it.
      kernelThread(k, helperStackTop(k, 'fsOpened'));
    }
    for (let cap = 1 << 20; ; cap *= 4) {
      const need = Math.ceil((SCRATCH + cap) / 65536) - mem.buffer.byteLength / 65536;
      if (need > 0) mem.grow(need);
      const n = k.k_fs_opened(BigInt(SCRATCH), cap);
      if (n === -34) continue;
      if (n < 0) throw new Error(`k_fs_opened: ${n}`);
      const text = new TextDecoder().decode(new Uint8Array(mem.buffer).slice(SCRATCH, SCRATCH + n));
      self.postMessage({ paths: text.split('\n').filter(Boolean) });
      return;
    }
  } catch (err) {
    self.postMessage({ error: String(err?.stack ?? err) });
  }
};
