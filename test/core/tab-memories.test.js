// The tab's wasm memories (lazy-fetch.mjs `wasmMemory`): a process's memory
// is counted while it runs and let go when it ends, and when the tab can
// hold no more (a browser reserves address space for each memory, and a tab
// runs out at about 124; `setTabMemoriesMax` stands in for that here) a new
// process ends as a crash that says so, and its parent goes on.
import { System } from '../../page/runtime/src/index.js';
import { MemoryFilesystem } from '../../page/runtime/src/fs.js';
import { setTabMemoriesMax, tabMemories } from '../../hosts/browser/js/lazy-fetch.mjs';

export const meta = {
  hosts: ['browser', 'node'], images: ['busybox'], timeout: 180,
  single: 'every process runs in the one kernel Worker there, whose memories the page does not count',
};

export default async function (t) {
  const { check, timeout } = t;
  const sys = new System({ baseUrl: t.baseUrl, debug: t.debug() });
  const fs = new MemoryFilesystem(sys);
  fs.addTar(t.work('bb.tar.zst'));
  const crashes = [];
  sys.addEventListener('crash', (e) => crashes.push({ pid: e.pid, name: e.name, reason: e.reason, code: e.code }));
  await timeout(sys.boot(), 60000, 'boot');
  const env = { PATH: '/bin:/usr/bin', HOME: '/root' };
  const run = (argv) => timeout(sys.run(argv, { env }), 60000, argv.join(' '));

  await run(['true']);
  const before = tabMemories(null, { reset: true });
  check('counted', before && before.live > 0, JSON.stringify(before));
  const r = await run(['sh', '-c', 'sh -c true; sh -c true; echo ran']);
  const after = tabMemories();
  t.say(`tab memories: ${before.live} live before, ${after.peak} at most, ${after.live} after`);
  check('a process memory is counted while it runs', r.stdout === 'ran\n' && after.peak > before.live, `${JSON.stringify(r)} ${JSON.stringify(after)}`);
  check('and let go when it ends', after.live <= before.live + 1, JSON.stringify(after));

  // the tab holds one more (an exited process's memory is reused, and the
  // pool's made ahead are taken first): of 12 children at once, some get none
  setTabMemoriesMax(after.live + 1);
  const full = await run(['sh', '-c', 'for i in 1 2 3 4 5 6 7 8 9 10 11 12; do sh -c "sleep 2" & done; wait; echo parent goes on']);
  setTabMemoriesMax(0);
  // (sh says "Killed" for each that got none)
  check('no memory: its parent goes on', full.stdout.endsWith('parent goes on\n'), JSON.stringify(full));
  for (let i = 0; i < 100 && !crashes.length; i++) await new Promise((r) => setTimeout(r, 50));
  const said = crashes.filter((c) => /the tab can hold no more process memories/.test(c.reason ?? ''));
  t.say(`${said.length} of 12 children got no memory`);
  check('a child with none ends as a crash, said plainly', said.length > 0 && said.every((c) => c.code === 137), JSON.stringify(crashes));

  const again = await run(['sh', '-c', 'sh -c "echo child ran"']);
  check('room again: a child runs', again.stdout === 'child ran\n', JSON.stringify(again));
  await sys.shutdown();
}
