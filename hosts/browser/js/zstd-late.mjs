// A zstddec decoder (page/vendor/zstddec, the package as published) whose
// instance, and the wasm memory it makes, come at its first decode. An
// engine reserves the same address space for any memory however small,
// and a tab has room for about 124: a decoder made in every Worker at its
// start (each thread of each process) ran pages out of them, though most
// threads never decode.
//
// The package's own init() fetches and instantiates asynchronously, and a
// decode here is synchronous. So the module is compiled at the Worker's
// start, and at the first decode it is instantiated synchronously and
// handed to the decoder the way init() hands it over (`_init`, with the
// instance).

/** A decoder from the zstddec module `ns` (zstddec-stream.mjs's exports)
 *  and its wasm at `wasmUrl`: `{ decode(bytes, size), decodeStreaming(parts) }`. */
export async function lateZstd(ns, wasmUrl) {
  const { ZSTDDecoder } = await ns;
  const url = new URL(wasmUrl);
  const bytes = typeof process !== 'undefined' && url.protocol === 'file:'
    ? await (await import('node:fs/promises')).readFile(url)
    : await fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: ${r.status}`);
      return r.arrayBuffer();
    });
  const mod = await WebAssembly.compile(bytes);
  let d = null;
  const now = () => {
    if (d) return d;
    const dec = new ZSTDDecoder();
    // the package's growth hook re-reads the memory; this one has it do so
    const inst = new WebAssembly.Instance(mod, { env: { emscripten_notify_memory_growth: () => dec._init({ instance: inst }) } });
    dec._init({ instance: inst });
    d = dec;
    return d;
  };
  return {
    decode: (bytes, size = 0) => now().decode(bytes, size),
    decodeStreaming: (parts, size) => now().decodeStreaming(parts, size),
  };
}
