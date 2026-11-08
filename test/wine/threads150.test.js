// 150 threads in one Wine process, in the browser: each thread is a Worker
// that instantiates Wine's modules, and the tab's wasm memories stay about
// one per process, not one per thread. Every memory costs a tab the same
// address space whatever its size (about 124 in all), so a memory per
// Worker (the late-call flags, a decoder's) ran a page out of them with a
// program's thread pool.
import { bootWine } from '../lib/wine.mjs';
import { tabMemories } from '../../hosts/browser/js/lazy-fetch.mjs';

export const meta = { hosts: ['browser'], arches: ['x86'], images: ['wine64', 'win-progs'], timeout: 900 };

export default async function (t) {
  const w = await bootWine(t);
  const before = tabMemories(null, { reset: true });
  const r = await w.run('threads150.exe', [], { timeout: 600000 });
  const after = tabMemories();
  t.say(`tab memories: ${before?.live} live before, ${after?.peak} at most with 150 threads, ${after?.live} after`);
  t.check('every thread starts', /^threads150 started 150 of 150$/m.test(r.stdout) && /^threads150 done$/m.test(r.stdout) && r.code === 0,
    `exit ${r.code}; stdout ${JSON.stringify(r.stdout.slice(-300))}; stderr ${JSON.stringify(r.stderr.trim().split('\n').slice(-4).join(' / '))}`);
  // The program's process and the few Wine starts beside it, not its threads.
  t.check('the memories follow processes, not threads', before && after && after.peak - before.live < 30, JSON.stringify({ before, after }));
  await w.sys.shutdown();
}
