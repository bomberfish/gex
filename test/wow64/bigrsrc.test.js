// A program whose resource section is 100 MiB (an icon, a version, then one
// big RCDATA: an installer's shape, as DirectX's redistributable is), and
// the same file cut short, as a program on the Desktop is seen while a drop
// still writes it. A shell reads its icon and version (rsrcread.c, an x64
// program, as explorer's desktop is): a second read of the icons, and the
// version, cost what they cost for the same program with a small resource
// section, natively and on the page. The cut file ends nothing: not the
// program (it says "done"), nor the wineserver (the image mapping it made of
// the file, whose version resource could not be read, is closed cleanly).
import { join } from 'node:path';
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { wow64 } from '../lib/winerun.mjs';

export const meta = { hosts: ['node'], arches: ['x86'], timeout: 1800 };

const SRC = new URL('../fixtures/wine/src/', import.meta.url).pathname;
const PAYLOAD = 100 << 20;
const CUT = 1 << 20;
/** How much more the big program's icon or version may cost than the small one's. */
const SLACK_MS = 250;

/** A 32x32 icon, 32 bits a pixel. */
function icon() {
  const px = 32 * 32 * 4, mask = 32 * 4, img = 40 + px + mask;
  const b = new Uint8Array(6 + 16 + img), v = new DataView(b.buffer);
  v.setUint16(2, 1, true); v.setUint16(4, 1, true);
  b[6] = 32; b[7] = 32; v.setUint16(10, 1, true); v.setUint16(12, 32, true);
  v.setUint32(14, img, true); v.setUint32(18, 22, true);
  v.setUint32(22, 40, true); v.setInt32(26, 32, true); v.setInt32(30, 64, true);
  v.setUint16(34, 1, true); v.setUint16(36, 32, true);
  for (let i = 0; i < px; i += 4) v.setUint32(62 + i, 0xff3070c0, true);
  return b;
}

const RC = (payload) => `1 ICON "icon.ico"
${payload ? '100 RCDATA "payload.bin"\n' : ''}1 VERSIONINFO
FILEVERSION 1,0,0,0
PRODUCTVERSION 1,0,0,0
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904b0"
    BEGIN
      VALUE "FileDescription", "Big resource section test"
      VALUE "FileVersion", "1.0"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
`;

const ms = (out, what) => Number(out.match(new RegExp(`^${what}[^\\n]*: (\\d+) ms$`, 'm'))?.[1] ?? NaN);

export default async function (t) {
  const w = wow64(t);
  const windres = w.tool('i686-w64-mingw32-windres');
  const reader = await w.mingw('rsrcread', { x64: true, libs: ['-lshell32', '-lversion'], src: join(SRC, 'rsrcread.c') });
  const dir = join(reader, '..');
  writeFileSync(join(dir, 'icon.ico'), icon());
  const payload = new Uint8Array(PAYLOAD);
  for (let i = 0; i < PAYLOAD; i += 4096) payload[i] = ((i >> 12) & 0xff) | 1;
  writeFileSync(join(dir, 'payload.bin'), payload);
  const files = { 'rsrcread.exe': reader };
  for (const [name, big] of [['small', false], ['big', true]]) {
    writeFileSync(join(dir, `${name}.rc`), RC(big));
    execFileSync(windres, [`${name}.rc`, '-O', 'coff', '-o', `${name}.res.o`], { cwd: dir });
    files[`${name}.exe`] = await w.mingw(name, { out: `${name}.exe`, src: join(SRC, 'bigimage.c'), libs: [join(dir, `${name}.res.o`)] });
  }
  writeFileSync(join(dir, 'cut.exe'), readFileSync(files['big.exe']).subarray(0, CUT));
  files['cut.exe'] = join(dir, 'cut.exe');

  for (const where of ['native', 'page']) {
    const read = (exe) => where === 'native'
      ? w.native(['C:\\rsrcread.exe', `C:\\${exe}`], { files, name: `n-${exe}`, timeout: 600000 })
      : w.page(['C:\\rsrcread.exe', `C:\\${exe}`], { pfxFiles: files, prefix: w.pagePrefix(), cwd: '/pfx/drive_c', name: `p-${exe}`, timeout: 600000 });
    const r = {};
    for (const exe of ['small.exe', 'big.exe', 'cut.exe']) {
      r[exe] = await read(exe);
      t.check(`${where} ${exe}: done`, /^done$/m.test(r[exe].out), w.detail(r[exe]));
      t.check(`${where} ${exe}: the wineserver lived`, !/door call died midway|trapped in a door call/.test(r[exe].out), w.detail(r[exe]));
    }
    for (const exe of ['small.exe', 'big.exe']) {
      t.check(`${where} ${exe}: its icon`, /^icons [1-9]\d* both:/m.test(r[exe].out), w.detail(r[exe]));
      t.check(`${where} ${exe}: its version`, /^version "Big resource section test":/m.test(r[exe].out), w.detail(r[exe]));
    }
    // the icons' second read (rsrcread.c), and the version's size
    for (const [what, line] of [['icons', 'icons \\d+ \\w+'], ['version', 'version size \\d+']]) {
      const [s, b] = [ms(r['small.exe'].out, line), ms(r['big.exe'].out, line)];
      t.say(`${where} ${what}: big ${b} ms, small ${s} ms`);
      t.check(`${where}: the big program's ${what} cost the small one's`, b <= s + SLACK_MS, `${b} ms against ${s} ms`);
    }
  }
  w.clean();
}
