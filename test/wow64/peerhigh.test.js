// A 64-bit parent reaching a suspended child's memory above 4 GiB
// (peerhigh.c: the PEB, then the image at 0x140000000: its header read,
// made writable, written and read back), as Firefox's and Thunderbird's sandbox does before it
// resumes each content process, on the page's memory64 twin (natively a
// fixed base there does not load). The server's own pointers stop at
// 4 GiB, so the child copies (an APC) for these.
import { wow64 } from '../lib/winerun.mjs';

export const meta = { hosts: ['node'], arches: ['x86'], timeout: 900 };

export default async function (t) {
  const w = wow64(t);
  // a fixed base at 0x140000000 (no relocations): the memory64 twin runs it
  const exe = await w.mingw('peerhigh', { flags: ['-Wl,--disable-dynamicbase', '-Wl,--disable-reloc-section'], libs: ['-lntdll'], out: 'peerhigh.exe', x64: true });
  const leg = (name, r) => {
    for (const m of r.out.matchAll(/^     (image base|page) (\S+)/gm)) t.say(`${name}: ${m[1]} ${m[2]}`);
    t.check(`${name}, peerhigh`, /^peerhigh done, 0 failed/m.test(r.out), w.detail(r, 4));
  };
  // 600 s: the twin's first start compiles its memory64 modules, which
  // took over 300 s under a full gate's load (about 270 s alone)
  leg('page', await w.page(['/t/peerhigh.exe'], { files: { '/t/peerhigh.exe': exe }, timeout: 600000 }));
  w.clean();
}
