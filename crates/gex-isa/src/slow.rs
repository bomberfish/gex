//! The AArch64 SIMD and crypto instructions with no good WebAssembly form, carried out on the
//! register file in plain Rust: saturating and doubling arithmetic, narrowing shifts, AES, SHA
//! and polynomial multiplies. The lifter sends a word it cannot express here (`gex-lift`), at the
//! cost of a host call; anything hot gets a proper lowering there and stops coming this way.
//!
//! Written from the Arm ARM's pseudocode, one family at a time. [`execute`] answers false for
//! anything else, so a word is only ever taken here when it is really understood.

/// The vector registers, as the 512 bytes of a context's `v0..v31`.
pub struct VRegs<'a>(pub &'a mut [u8]);

impl VRegs<'_> {
    fn get(&self, n: u32) -> u128 {
        let at = n as usize * 16;
        u128::from_le_bytes(self.0[at..at + 16].try_into().unwrap())
    }
    fn set(&mut self, n: u32, v: u128) {
        let at = n as usize * 16;
        self.0[at..at + 16].copy_from_slice(&v.to_le_bytes());
    }
}

const QC: u32 = 1 << 27; // FPSR's cumulative saturation bit

fn bits(raw: u32, hi: u32, lo: u32) -> u32 {
    (raw >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

fn lane(v: u128, i: u32, bytes: u32) -> u64 {
    let w = bytes * 8;
    ((v >> (i * w)) as u64) & if w == 64 { u64::MAX } else { (1u64 << w) - 1 }
}

fn slane(v: u128, i: u32, bytes: u32) -> i64 {
    let w = bytes * 8;
    let x = lane(v, i, bytes);
    if w == 64 { x as i64 } else { ((x << (64 - w)) as i64) >> (64 - w) }
}

fn put(v: &mut u128, i: u32, bytes: u32, x: u64) {
    let w = bytes * 8;
    let m: u128 = if w == 64 { u64::MAX as u128 } else { (1u128 << w) - 1 };
    *v = (*v & !(m << (i * w))) | (((x as u128) & m) << (i * w));
}

/// `x` clamped to a signed `bytes`-wide lane, noting saturation.
fn sat_s(x: i128, bytes: u32, fpsr: &mut u32) -> u64 {
    let w = bytes * 8;
    let (lo, hi) = (-(1i128 << (w - 1)), (1i128 << (w - 1)) - 1);
    if x < lo || x > hi {
        *fpsr |= QC;
    }
    x.clamp(lo, hi) as u64
}

/// `x` clamped to an unsigned `bytes`-wide lane, noting saturation.
fn sat_u(x: i128, bytes: u32, fpsr: &mut u32) -> u64 {
    let hi = (1i128 << (bytes * 8)) - 1;
    if x < 0 || x > hi {
        *fpsr |= QC;
    }
    x.clamp(0, hi) as u64
}

fn f32_of(x: u64) -> f32 {
    f32::from_bits(x as u32)
}

/// A half-precision value from a single, rounding to nearest even (FPCR's default mode).
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let frac = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if frac != 0 { 0x200 | (frac >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    let (mant, shift) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        (frac | 0x80_0000, (14 - e) as u32)
    } else {
        (frac, 13)
    };
    let mut m = mant >> shift;
    let rem = mant & ((1 << shift) - 1);
    let half = 1u32 << (shift - 1);
    if rem > half || (rem == half && m & 1 == 1) {
        m += 1;
    }
    let base = if e <= 0 { 0 } else { (e as u32) << 10 };
    sign | (base + m) as u16
}

fn f64_to_f16(v: f64) -> u16 {
    // Through single would round twice. Single rounded to odd instead -- truncated toward zero,
    // its last bit set when anything was cut off -- keeps what a double holds below single's
    // bits sticky, and single has the 2 * 11 + 2 bits that makes half's one rounding of it exact.
    // (`as f32` rounds to nearest, which can go up past `v`: hence the step back first.)
    if v.is_nan() || v.is_infinite() {
        return f32_to_f16(v as f32);
    }
    let mut t = v as f32;
    if (t as f64).abs() > v.abs() {
        t = f32::from_bits(t.to_bits() - 1);
    }
    if t as f64 != v {
        t = f32::from_bits(t.to_bits() | 1);
    }
    f32_to_f16(t)
}

fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let frac = (h & 0x3ff) as f64;
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        0x1f if frac == 0.0 => sign * f64::INFINITY,
        0x1f => f64::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp as i32 - 15),
    }
}

// ---- AES ------------------------------------------------------------------

const fn aes_sboxes() -> ([u8; 256], [u8; 256]) {
    let (mut s, mut inv) = ([0u8; 256], [0u8; 256]);
    let (mut p, mut q) = (1u8, 1u8);
    loop {
        // p walks the field's non-zero elements (times 3), q its inverses (divided by 3).
        p = p ^ (p << 1) ^ if p & 0x80 != 0 { 0x1b } else { 0 };
        q ^= q << 1;
        q ^= q << 2;
        q ^= q << 4;
        if q & 0x80 != 0 {
            q ^= 0x09;
        }
        let x = q ^ q.rotate_left(1) ^ q.rotate_left(2) ^ q.rotate_left(3) ^ q.rotate_left(4) ^ 0x63;
        s[p as usize] = x;
        inv[x as usize] = p;
        if p == 1 {
            break;
        }
    }
    s[0] = 0x63;
    inv[0x63] = 0;
    (s, inv)
}
const AES: ([u8; 256], [u8; 256]) = aes_sboxes();

fn xt(a: u8) -> u8 {
    (a << 1) ^ if a & 0x80 != 0 { 0x1b } else { 0 }
}
fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut r = 0;
    while b != 0 {
        if b & 1 != 0 {
            r ^= a;
        }
        a = xt(a);
        b >>= 1;
    }
    r
}

fn aes_sub_shift(v: u128, inverse: bool) -> u128 {
    let b = v.to_le_bytes();
    let mut o = [0u8; 16];
    for c in 0..4 {
        for r in 0..4 {
            let (to, from) = if inverse { (r + ((c + r) % 4) * 4, r + c * 4) } else { (r + c * 4, r + ((c + r) % 4) * 4) };
            o[to] = if inverse { AES.1[b[from] as usize] } else { AES.0[b[from] as usize] };
        }
    }
    u128::from_le_bytes(o)
}

fn aes_mix(v: u128, inverse: bool) -> u128 {
    let b = v.to_le_bytes();
    let k: [u8; 4] = if inverse { [14, 11, 13, 9] } else { [2, 3, 1, 1] };
    let mut o = [0u8; 16];
    for c in 0..4 {
        for r in 0..4 {
            o[c * 4 + r] = (0..4).fold(0, |x, i| x ^ gmul(b[c * 4 + (r + i) % 4], k[i]));
        }
    }
    u128::from_le_bytes(o)
}

// ---- SHA ------------------------------------------------------------------

fn w32(v: u128, i: u32) -> u32 {
    lane(v, i, 4) as u32
}
fn from_w32(w: [u32; 4]) -> u128 {
    (w[0] as u128) | (w[1] as u128) << 32 | (w[2] as u128) << 64 | (w[3] as u128) << 96
}

fn sha1_rounds(kind: u32, abcd: u128, e: u32, wk: u128) -> u128 {
    let mut s = [w32(abcd, 0), w32(abcd, 1), w32(abcd, 2), w32(abcd, 3)];
    let mut e = e;
    for i in 0..4 {
        let (b, c, d) = (s[1], s[2], s[3]);
        let f = match kind {
            0 => (b & c) | (!b & d),             // choose
            1 => b ^ c ^ d,                      // parity
            _ => (b & c) | (b & d) | (c & d),    // majority
        };
        let t = e.wrapping_add(s[0].rotate_left(5)).wrapping_add(f).wrapping_add(w32(wk, i));
        e = d;
        s = [t, s[0], b.rotate_left(30), c];
    }
    from_w32(s)
}

fn sha256_rounds(x: u128, y: u128, wk: u128, part1: bool) -> u128 {
    let (mut s0, mut s1) = (x, y);
    for i in 0..4 {
        let [a, b, c, d] = [w32(s0, 0), w32(s0, 1), w32(s0, 2), w32(s0, 3)];
        let [e, f, g, h] = [w32(s1, 0), w32(s1, 1), w32(s1, 2), w32(s1, 3)];
        let ch = (e & f) ^ (!e & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let t = h.wrapping_add(big1).wrapping_add(ch).wrapping_add(w32(wk, i));
        let new_e = d.wrapping_add(t);
        let new_a = t.wrapping_add(big0).wrapping_add(maj);
        s1 = from_w32([new_e, e, f, g]);
        s0 = from_w32([new_a, a, b, c]);
    }
    if part1 { s0 } else { s1 }
}

// ---- Polynomial multiply --------------------------------------------------

fn clmul(a: u64, b: u64) -> u128 {
    (0..64).fold(0u128, |r, i| if b >> i & 1 != 0 { r ^ ((a as u128) << i) } else { r })
}

// ---- The instructions -----------------------------------------------------

/// The system register number (`gex_sysreg_write`) that asks the host to carry out the word it
/// is given: the lifter's way here, since it flushes and reloads the cached guest registers.
pub const SYSREG: i32 = -201;

/// Whether [`execute`] takes this word. Nothing else is touched: it runs on scratch registers.
pub fn known(raw: u32) -> bool {
    let mut scratch = [0u8; 512];
    execute(raw, &mut VRegs(&mut scratch), &mut 0)
}

/// Carry out `raw` on the registers. False when it is not one of this module's, nothing changed.
pub fn execute(raw: u32, v: &mut VRegs, fpsr: &mut u32) -> bool {
    let (rd, rn, rm) = (raw & 31, bits(raw, 9, 5), bits(raw, 20, 16));
    let q = raw >> 30 & 1 == 1;
    let u = raw >> 29 & 1 == 1;
    let size = bits(raw, 23, 22);
    let bytes = 1u32 << size;
    let nb = if q { 16 } else { 8 };
    let lanes = nb / bytes;
    let high0 = |x: u128| if q { x } else { x & u64::MAX as u128 };

    // AES: 0100 1110 0010 1000 0 opcode 10 Rn Rd
    if raw & 0xffff_cc00 == 0x4e28_4800 {
        let d = v.get(rd);
        let n = v.get(rn);
        let r = match bits(raw, 13, 12) {
            0 => aes_sub_shift(d ^ n, false), // aese
            1 => aes_sub_shift(d ^ n, true),  // aesd
            2 => aes_mix(n, false),           // aesmc
            _ => aes_mix(n, true),            // aesimc
        };
        v.set(rd, r);
        return true;
    }
    // SHA, three registers: 0101 1110 000 Rm 0 opc 00 Rn Rd
    if raw & 0xffe0_8c00 == 0x5e00_0000 {
        let (d, n, m) = (v.get(rd), v.get(rn), v.get(rm));
        let r = match bits(raw, 14, 12) {
            0 => sha1_rounds(0, d, w32(n, 0), m),
            1 => sha1_rounds(1, d, w32(n, 0), m),
            2 => sha1_rounds(2, d, w32(n, 0), m),
            3 => (d >> 64 | n << 64) ^ d ^ m, // sha1su0: {n<1:0>, d<3:2>} ^ d ^ m
            4 => sha256_rounds(d, n, m, true),
            5 => sha256_rounds(n, d, m, false),
            6 => sha256su1(d, n, m),
            _ => return false,
        };
        v.set(rd, r);
        return true;
    }
    // SHA, two registers: 0101 1110 0010 1000 0 opc 10 Rn Rd
    if raw & 0xffff_8c00 == 0x5e28_0800 {
        let (d, n) = (v.get(rd), v.get(rn));
        let r = match bits(raw, 14, 12) {
            0 => w32(n, 0).rotate_left(30) as u128, // sha1h
            1 => {
                // sha1su1
                let t = d ^ (n >> 32);
                let mut o = [t as u32, (t >> 32) as u32, (t >> 64) as u32, (t >> 96) as u32];
                for x in &mut o {
                    *x = x.rotate_left(1);
                }
                o[3] ^= (t as u32).rotate_left(2);
                from_w32(o)
            }
            2 => {
                // sha256su0
                let s = |x: u32| x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
                let w = [w32(d, 0), w32(d, 1), w32(d, 2), w32(d, 3), w32(n, 0)];
                from_w32([
                    w[0].wrapping_add(s(w[1])),
                    w[1].wrapping_add(s(w[2])),
                    w[2].wrapping_add(s(w[3])),
                    w[3].wrapping_add(s(w[4])),
                ])
            }
            _ => return false,
        };
        v.set(rd, r);
        return true;
    }
    // fcvt to and from half: 0001 1110 ftype 1 0001 opc 10000 Rn Rd, scalar
    if raw & 0xff3e_7c00 == 0x1e22_4000 {
        let ftype = size;
        let opc = bits(raw, 16, 15);
        let n = v.get(rn);
        let r: u128 = match (ftype, opc) {
            (0, 3) => f32_to_f16(f32_of(n as u64)) as u128,                    // s -> h
            (1, 3) => f64_to_f16(f64::from_bits(n as u64)) as u128,            // d -> h
            (3, 0) => (f16_to_f64(n as u16) as f32).to_bits() as u128,         // h -> s
            (3, 1) => f16_to_f64(n as u16).to_bits() as u128,                  // h -> d
            _ => return false,
        };
        v.set(rd, r);
        return true;
    }
    // Scalar saturating add and subtract: 01 U 11110 size 1 Rm op 1 Rn Rd
    if raw & 0xdf20_0400 == 0x5e20_0400 {
        let op = bits(raw, 15, 11);
        if op != 0b00001 && op != 0b00101 {
            return false;
        }
        let (n, m) = (v.get(rn), v.get(rm));
        let out = if u {
            let (a, b) = (lane(n, 0, bytes) as i128, lane(m, 0, bytes) as i128);
            sat_u(if op == 1 { a + b } else { a - b }, bytes, fpsr)
        } else {
            let (a, b) = (slane(n, 0, bytes) as i128, slane(m, 0, bytes) as i128);
            sat_s(if op == 1 { a + b } else { a - b }, bytes, fpsr)
        };
        let mut r = 0u128;
        put(&mut r, 0, bytes, out);
        v.set(rd, r);
        return true;
    }
    // Scalar saturating narrowing shifts: 01 U 111110 immh immb op 1 Rn Rd
    if raw & 0xdf80_0400 == 0x5f00_0400 && bits(raw, 22, 19) != 0 {
        let immh = bits(raw, 22, 19);
        let esize = 8u32 << (31 - immh.leading_zeros());
        let op = bits(raw, 15, 11);
        if esize == 64 || !(0b10000..=0b10011).contains(&op) || (!u && op & 2 == 0) {
            return false;
        }
        let mut r = 0u128;
        let out = narrow_lane(v.get(rn), 0, esize / 8, u, op, 2 * esize - bits(raw, 22, 16), fpsr);
        put(&mut r, 0, esize / 8, out);
        v.set(rd, r);
        return true;
    }
    // Everything below is the 0 Q U 0111x family.
    if raw >> 31 != 0 || bits(raw, 28, 25) != 0b0111 {
        return false;
    }

    // Shift by immediate: 0 Q U 011110 immh immb opcode 1 Rn Rd
    if bits(raw, 28, 23) == 0b011110 && raw >> 10 & 1 == 1 && bits(raw, 22, 19) != 0 {
        let immh = bits(raw, 22, 19);
        let imm = bits(raw, 22, 16);
        let esize = 8u32 << (31 - immh.leading_zeros());
        let eb = esize / 8;
        let op = bits(raw, 15, 11);
        let n = v.get(rn);
        match op {
            // narrowing right shifts: Vd narrow, Vn wide, `q` picks the upper half
            0b10000 | 0b10001 | 0b10010 | 0b10011 => {
                if esize == 64 {
                    return false;
                }
                let shift = 2 * esize - imm;
                if !u && op & 2 == 0 {
                    return false; // shrn, rshrn: the lifter's
                }
                let mut r = if q { v.get(rd) & u64::MAX as u128 } else { 0 };
                let base = if q { 8 / eb } else { 0 };
                for i in 0..8 / eb {
                    put(&mut r, base + i, eb, narrow_lane(n, i, eb, u, op, shift, fpsr));
                }
                v.set(rd, r);
                true
            }
            // sqshl, uqshl, sqshlu by immediate
            0b01110 | 0b01100 => {
                if !q && esize == 64 {
                    return false;
                }
                let shift = imm - esize;
                let mut r = 0u128;
                for i in 0..nb / eb {
                    let out = if op == 0b01100 {
                        if !u {
                            return false;
                        }
                        sat_u((slane(n, i, eb) as i128) << shift, eb, fpsr) // sqshlu
                    } else if u {
                        sat_u((lane(n, i, eb) as i128) << shift, eb, fpsr) // uqshl
                    } else {
                        sat_s((slane(n, i, eb) as i128) << shift, eb, fpsr) // sqshl
                    };
                    put(&mut r, i, eb, out);
                }
                v.set(rd, r);
                true
            }
            _ => false,
        }
    } else if bits(raw, 28, 24) == 0b01111 && raw >> 10 & 1 == 0 {
        // By element: 0 Q U 01111 size L M Rm opcode H 0 Rn Rd
        if size == 0 || size == 3 {
            return false;
        }
        let (l, m_bit, h) = (raw >> 21 & 1, raw >> 20 & 1, raw >> 11 & 1);
        let (index, rm_e) = if size == 1 { (h << 2 | l << 1 | m_bit, rm & 15) } else { (h << 1 | l, rm) };
        let e = slane(v.get(rm_e), index, bytes) as i128;
        let op = bits(raw, 15, 12);
        let n = v.get(rn);
        match (op, u) {
            (0b1100, false) | (0b1101, false) => {
                let round: i128 = if op == 0b1101 { 1i128 << (bytes * 8 - 1) } else { 0 };
                let mut r = 0u128;
                for i in 0..lanes {
                    let x = (2 * slane(n, i, bytes) as i128 * e + round) >> (bytes * 8);
                    put(&mut r, i, bytes, sat_s(x, bytes, fpsr));
                }
                v.set(rd, r);
                true
            }
            // sqdmlal, sqdmlal2, sqdmlsl, sqdmlsl2 by element
            (0b0011, false) | (0b0111, false) => {
                let base = if q { lanes / 2 } else { 0 };
                let d = v.get(rd);
                let mut r = 0u128;
                for i in 0..8 / bytes {
                    let p = sat_s(2 * slane(n, base + i, bytes) as i128 * e, bytes * 2, fpsr) as i64 as i128;
                    let acc = slane(d, i, bytes * 2) as i128;
                    let sum = if op == 0b0011 { acc + p } else { acc - p };
                    put(&mut r, i, bytes * 2, sat_s(sum, bytes * 2, fpsr));
                }
                v.set(rd, r);
                true
            }
            _ => false,
        }
    } else if bits(raw, 28, 24) == 0b01110 && raw >> 21 & 1 == 1 {
        // Three different: ... 1 Rm opcode 00 Rn Rd
        if bits(raw, 11, 10) == 0b00 {
            let op = bits(raw, 15, 12);
            let (n, m) = (v.get(rn), v.get(rm));
            return match (op, u) {
                // pmull, pmull2
                (0b1110, false) if size == 3 => {
                    let i = q as u32;
                    v.set(rd, clmul(lane(n, i, 8), lane(m, i, 8)));
                    true
                }
                (0b1110, false) if size == 0 => {
                    let base = if q { 8 } else { 0 };
                    let mut r = 0u128;
                    for i in 0..8 {
                        put(&mut r, i, 2, clmul(lane(n, base + i, 1), lane(m, base + i, 1)) as u64);
                    }
                    v.set(rd, r);
                    true
                }
                // sqdmlal, sqdmlsl, sqdmull and their upper forms
                (0b1001, false) | (0b1011, false) | (0b1101, false) if size == 1 || size == 2 => {
                    let base = if q { lanes / 2 } else { 0 };
                    let d = v.get(rd);
                    let mut r = 0u128;
                    for i in 0..8 / bytes {
                        let p = sat_s(2 * slane(n, base + i, bytes) as i128 * slane(m, base + i, bytes) as i128, bytes * 2, fpsr) as i64 as i128;
                        let out = match op {
                            0b1101 => p,
                            0b1001 => slane(d, i, bytes * 2) as i128 + p,
                            _ => slane(d, i, bytes * 2) as i128 - p,
                        };
                        put(&mut r, i, bytes * 2, sat_s(out, bytes * 2, fpsr));
                    }
                    v.set(rd, r);
                    true
                }
                _ => false,
            };
        }
        // Three same: ... 1 Rm opcode 1 Rn Rd
        if raw >> 10 & 1 == 1 {
            let op = bits(raw, 15, 11);
            let (n, m) = (v.get(rn), v.get(rm));
            if !q && size == 3 && op < 0b11000 {
                return false;
            }
            let mut r = 0u128;
            match (op, u) {
                (0b00001, _) | (0b00101, _) => {
                    // sqadd, uqadd, sqsub, uqsub
                    for i in 0..lanes {
                        let out = if u {
                            let (a, b) = (lane(n, i, bytes) as i128, lane(m, i, bytes) as i128);
                            sat_u(if op == 1 { a + b } else { a - b }, bytes, fpsr)
                        } else {
                            let (a, b) = (slane(n, i, bytes) as i128, slane(m, i, bytes) as i128);
                            sat_s(if op == 1 { a + b } else { a - b }, bytes, fpsr)
                        };
                        put(&mut r, i, bytes, out);
                    }
                }
                (0b01010, _) | (0b01011, _) | (0b01001, _) => {
                    // srshl, urshl; sqrshl, uqrshl; sqshl, uqshl by register
                    let rounding = op != 0b01001;
                    let saturating = op != 0b01010;
                    for i in 0..lanes {
                        let s = (lane(m, i, bytes) as u8) as i8 as i32;
                        let a: i128 = if u { lane(n, i, bytes) as i128 } else { slane(n, i, bytes) as i128 };
                        let x: i128 = if s >= 64 {
                            // Past any lane: nothing is left of a plain shift, and a saturating
                            // one saturates unless the lane is zero. (<< 64 would not fit i128.)
                            if !saturating { 0 } else { a.signum() << 100 }
                        } else if s >= 0 {
                            a << s
                        } else {
                            let sh = (-s).min(100);
                            let round = if rounding { 1i128 << (sh - 1) } else { 0 };
                            (a + round) >> sh
                        };
                        let out = match (saturating, u) {
                            (true, true) => sat_u(x, bytes, fpsr),
                            (true, false) => sat_s(x, bytes, fpsr),
                            (false, _) => x as u64,
                        };
                        put(&mut r, i, bytes, out);
                    }
                }
                (0b10110, _) => {
                    // sqdmulh, sqrdmulh
                    if size == 0 || size == 3 {
                        return false;
                    }
                    let round: i128 = if u { 1i128 << (bytes * 8 - 1) } else { 0 };
                    for i in 0..lanes {
                        let x = (2 * slane(n, i, bytes) as i128 * slane(m, i, bytes) as i128 + round) >> (bytes * 8);
                        put(&mut r, i, bytes, sat_s(x, bytes, fpsr));
                    }
                }
                _ => return false,
            }
            v.set(rd, high0(r));
            return true;
        }
        // Two-register misc: ... 10000 opcode 10 Rn Rd
        if bits(raw, 21, 17) == 0b10000 && bits(raw, 11, 10) == 0b10 {
            let op = bits(raw, 16, 12);
            let n = v.get(rn);
            return match (op, u) {
                // saddlp, sadalp, uaddlp, uadalp on word lanes
                (0b00010, _) | (0b00110, _) if size == 2 => {
                    let d = v.get(rd);
                    let mut r = 0u128;
                    for i in 0..nb / 8 {
                        let (a, b): (u64, u64) = if u {
                            (lane(n, i * 2, 4), lane(n, i * 2 + 1, 4))
                        } else {
                            (slane(n, i * 2, 4) as u64, slane(n, i * 2 + 1, 4) as u64)
                        };
                        let acc = if op == 0b00110 { lane(d, i, 8) } else { 0 };
                        put(&mut r, i, 8, a.wrapping_add(b).wrapping_add(acc));
                    }
                    v.set(rd, r);
                    true
                }
                _ => false,
            };
        }
        // fneg, fabs on half-precision lanes: the FP16 group, 1 1 11100 01111 10
        if size == 3 && bits(raw, 21, 17) == 0b11100 && bits(raw, 16, 12) == 0b01111 && bits(raw, 11, 10) == 0b10 {
            let n = v.get(rn);
            let mut r = 0u128;
            for i in 0..nb / 2 {
                let x = lane(n, i, 2);
                put(&mut r, i, 2, if u { x ^ 0x8000 } else { x & 0x7fff });
            }
            v.set(rd, r);
            return true;
        }
        false
    } else {
        false
    }
}

/// A wide lane shifted right and saturated into a lane of `eb` bytes: `sqshrn`, `sqrshrn`,
/// `uqshrn`, `uqrshrn`, `sqshrun`, `sqrshrun` by their `U` and `opcode`.
fn narrow_lane(n: u128, i: u32, eb: u32, u: bool, op: u32, shift: u32, fpsr: &mut u32) -> u64 {
    let unsigned_in = u && op & 2 != 0;
    let wide = if unsigned_in { lane(n, i, eb * 2) as i128 } else { slane(n, i, eb * 2) as i128 };
    let round = if op & 1 == 1 { 1i128 << (shift - 1) } else { 0 };
    let shifted = (wide + round) >> shift;
    if u { sat_u(shifted, eb, fpsr) } else { sat_s(shifted, eb, fpsr) }
}

/// T0 is unused by the upper half's source: the first two elements come from `m`'s upper half,
/// the last two from this result's first two, each with `d` and `m<31:0> : n<127:32>` added.
fn sha256su1(d: u128, n: u128, m: u128) -> u128 {
    let sigma1 = |x: u32| x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10);
    let t = [w32(n, 1), w32(n, 2), w32(n, 3), w32(m, 0)];
    let high = [w32(m, 2), w32(m, 3)];
    let mut o = [0u32; 4];
    for i in 0..4 {
        let source = if i < 2 { high[i] } else { o[i - 2] };
        o[i] = sigma1(source).wrapping_add(w32(d, i as u32)).wrapping_add(t[i]);
    }
    from_w32(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(raw: u32, regs: &mut [u128; 32]) -> u32 {
        let mut bytes = [0u8; 512];
        for (i, r) in regs.iter().enumerate() {
            bytes[i * 16..i * 16 + 16].copy_from_slice(&r.to_le_bytes());
        }
        let mut fpsr = 0;
        let mut v = VRegs(&mut bytes);
        assert!(execute(raw, &mut v, &mut fpsr), "{raw:08x} was not taken");
        for (i, r) in regs.iter_mut().enumerate() {
            *r = u128::from_le_bytes(bytes[i * 16..i * 16 + 16].try_into().unwrap());
        }
        fpsr
    }

    fn zeroed() -> [u128; 32] {
        [0; 32]
    }

    /// FIPS-197 appendix A.1's cipher example, by `aese` and `aesmc`: the whole block.
    #[test]
    fn aes_encrypts_the_fips_example() {
        // Round keys of key 2b7e151628aed2a6abf7158809cf4f3c, expanded here by the book.
        let key = 0x3c4fcf098815f7aba6d2ae2816157e2bu128;
        let mut rk = vec![key];
        let mut rcon = 1u8;
        for _ in 0..10 {
            let p = *rk.last().unwrap();
            let w3 = w32(p, 3).rotate_right(8);
            let sub = w3.to_le_bytes().map(|b| AES.0[b as usize]);
            let t = u32::from_le_bytes(sub) ^ rcon as u32;
            rcon = xt(rcon);
            let w0 = w32(p, 0) ^ t;
            let w1 = w32(p, 1) ^ w0;
            let w2 = w32(p, 2) ^ w1;
            let w3b = w32(p, 3) ^ w2;
            rk.push(from_w32([w0, w1, w2, w3b]));
        }
        let mut regs = zeroed();
        regs[0] = 0x340737e0a29831318d305a88a8f64332; // the example's plaintext 3243f6a8885a308d313198a2e0370734
        let aese_v0_v1 = 0x4e284820u32;
        let aesmc_v0_v0 = 0x4e286800u32;
        for k in rk.iter().take(9) {
            regs[1] = *k;
            run(aese_v0_v1, &mut regs);
            run(aesmc_v0_v0, &mut regs);
        }
        regs[1] = rk[9];
        run(aese_v0_v1, &mut regs);
        assert_eq!(regs[0] ^ rk[10], 0x320b6a19978511dcfb09dc021d842539); // 3925841d02dc09fbdc118597196a0b32
    }

    /// `aesd` and `aesimc` undo it: the same block back.
    #[test]
    fn aes_decrypts_what_it_encrypted() {
        let mut regs = zeroed();
        let start = 0x0123456789abcdeffedcba9876543210u128;
        regs[0] = start;
        run(0x4e284820, &mut regs); // aese v0, v1 (key 0)
        run(0x4e285820, &mut regs); // aesd v0, v1
        assert_eq!(regs[0], start);
        run(0x4e286800, &mut regs); // aesmc v0, v0
        run(0x4e287800, &mut regs); // aesimc v0, v0
        assert_eq!(regs[0], start);
    }

    /// One whole SHA-256 block of "abc", by the instructions, against the known digest.
    #[test]
    fn sha256_digests_abc() {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let mut block = [0u8; 64];
        block[..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[63] = 24;
        let mut w: Vec<u128> = block.chunks(16).map(|c| {
            from_w32([0, 4, 8, 12].map(|i| u32::from_be_bytes(c[i..i + 4].try_into().unwrap())))
        }).collect();
        let init = [0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
        let mut regs = zeroed();
        regs[0] = from_w32([init[0], init[1], init[2], init[3]]);
        regs[1] = from_w32([init[4], init[5], init[6], init[7]]);
        for round in 0..16 {
            let i = round % 4;
            let wk = from_w32([0, 1, 2, 3].map(|k| w32(w[i], k).wrapping_add(K[round * 4 + k as usize])));
            regs[2] = regs[0];
            regs[4] = wk;
            run(0x5e044020, &mut regs); // sha256h q0, q1, v4.4s
            run(0x5e045041, &mut regs); // sha256h2 q1, q2, v4.4s
            if round < 12 {
                // the next schedule words: su0 over w[i], w[i+1]; su1 over that, w[i+2], w[i+3]
                regs[5] = w[i];
                regs[6] = w[(i + 1) % 4];
                regs[7] = w[(i + 2) % 4];
                regs[8] = w[(i + 3) % 4];
                run(0x5e2828c5, &mut regs); // sha256su0 v5.4s, v6.4s
                run(0x5e0860e5, &mut regs); // sha256su1 v5.4s, v7.4s, v8.4s
                w[i] = regs[5];
            }
        }
        let digest = [
            w32(regs[0], 0).wrapping_add(init[0]), w32(regs[0], 1).wrapping_add(init[1]),
            w32(regs[0], 2).wrapping_add(init[2]), w32(regs[0], 3).wrapping_add(init[3]),
            w32(regs[1], 0).wrapping_add(init[4]), w32(regs[1], 1).wrapping_add(init[5]),
            w32(regs[1], 2).wrapping_add(init[6]), w32(regs[1], 3).wrapping_add(init[7]),
        ];
        assert_eq!(digest, [0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad]);
    }

    /// SHA-1's rounds and schedule, one whole block of "abc", against the known digest.
    #[test]
    fn sha1_digests_abc() {
        let mut block = [0u8; 64];
        block[..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[63] = 24;
        let mut w: Vec<u128> = block.chunks(16).map(|c| {
            from_w32([0, 4, 8, 12].map(|i| u32::from_be_bytes(c[i..i + 4].try_into().unwrap())))
        }).collect();
        let init = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];
        let mut regs = zeroed();
        regs[0] = from_w32([init[0], init[1], init[2], init[3]]);
        let mut e = init[4];
        for group in 0..20 {
            let i = group % 4;
            let k = [0x5a827999u32, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6][group / 5];
            let wk = from_w32([0, 1, 2, 3].map(|j| w32(w[i], j).wrapping_add(k)));
            regs[1] = e as u128;
            regs[2] = wk;
            regs[3] = regs[0];
            run(0x5e280860, &mut regs); // sha1h s0, s3: the next e, from this a
            let next_e = w32(regs[0], 0);
            regs[0] = regs[3];
            let op = [0x5e020020u32, 0x5e021020, 0x5e022020, 0x5e021020][group / 5]; // sha1c/p/m/p q0, s1, v2
            run(op, &mut regs);
            e = next_e;
            if group < 16 {
                regs[5] = w[i];
                regs[6] = w[(i + 1) % 4];
                regs[7] = w[(i + 2) % 4];
                regs[8] = w[(i + 3) % 4];
                run(0x5e0730c5, &mut regs); // sha1su0 v5, v6, v7
                run(0x5e281905, &mut regs); // sha1su1 v5, v8
                w[i] = regs[5];
            }
        }
        let got = [
            w32(regs[0], 0).wrapping_add(init[0]), w32(regs[0], 1).wrapping_add(init[1]),
            w32(regs[0], 2).wrapping_add(init[2]), w32(regs[0], 3).wrapping_add(init[3]),
            e.wrapping_add(init[4]),
        ];
        assert_eq!(got, [0xa9993e36, 0x4706816a, 0xba3e2571, 0x7850c26c, 0x9cd0d89d]);
    }

    #[test]
    fn saturating_doubling_multiply() {
        let mut regs = zeroed();
        // sqrdmulh v1.4s, v1.4s, v0.s[0]: 0.5 * 0.5 in Q31, and -1 * -1 saturating with QC.
        regs[0] = 0x4000_0000;
        regs[1] = 0x4000_0000;
        assert_eq!(run(0x4f80d021, &mut regs) & QC, 0);
        assert_eq!(w32(regs[1], 0), 0x2000_0000);
        regs[0] = 0x8000_0000;
        regs[1] = 0x8000_0000;
        assert_ne!(run(0x4f80d021, &mut regs) & QC, 0);
        assert_eq!(w32(regs[1], 0), 0x7fff_ffff);
        // sqdmulh v4.4h, v4.4h, v0.4h truncates and clears the upper half.
        regs[0] = 0x4000;
        regs[4] = (0xffffu128 << 64) | 0x4000;
        run(0x0e60b484, &mut regs);
        assert_eq!(regs[4], 0x2000);
    }

    #[test]
    fn saturating_add_and_subtract_word_lanes() {
        let mut regs = zeroed();
        regs[0] = 0x7fff_ffff;
        regs[1] = 1;
        run(0x0ea00c20, &mut regs); // sqadd v0.2s, v1.2s, v0.2s
        assert_eq!(w32(regs[0], 0), 0x7fff_ffff);
        regs[1] = 0x8000_0000;
        regs[2] = 1;
        run(0x0ea12c42, &mut regs); // sqsub v2.2s, v2.2s, v1.2s: 1 - INT_MIN
        assert_eq!(w32(regs[2], 0), 0x7fff_ffff);
    }

    #[test]
    fn narrowing_shifts() {
        let mut regs = zeroed();
        // sqrshrn v28.8b, v2.8h, #2: (5 + 2) >> 2 = 1; 1000 saturates to 127; -1000 to -128.
        regs[2] = 5 | (1000u128) << 16 | ((-1000i16 as u16 as u128) << 32);
        run(0x0f0e9c5c, &mut regs);
        assert_eq!(regs[28], 0x80_7f_01);
        // sqrshrn2 v28.16b, v6.8h, #2 keeps the lower half.
        regs[28] = 0x11;
        regs[6] = 8;
        run(0x4f0e9cdc, &mut regs);
        assert_eq!(regs[28], (2u128 << 64) | 0x11);
        // sqshrn v28.8b, v16.8h, #5 truncates.
        regs[16] = 63;
        run(0x0f0b961c, &mut regs);
        assert_eq!(regs[28], 1);
    }

    #[test]
    fn scalar_saturating_forms() {
        let mut regs = zeroed();
        // sqadd s0, s0, s1 clears the rest of the register.
        regs[0] = (5u128 << 64) | 0x7fff_ffff;
        regs[1] = 1;
        assert_ne!(run(0x5ea10c00, &mut regs) & QC, 0);
        assert_eq!(regs[0], 0x7fff_ffff);
        // sqsub s0, s0, s1
        regs[0] = 10;
        regs[1] = 3;
        run(0x5ea12c00, &mut regs);
        assert_eq!(regs[0], 7);
        // sqrshrn h26, s26, #15: 1.0 in Q31 less a little, to Q15, rounded and saturated.
        regs[26] = 0x7fff_ffff;
        run(0x5f119f5a, &mut regs);
        assert_eq!(regs[26], 0x7fff);
        regs[26] = 0x0001_8000;
        run(0x5f119f5a, &mut regs);
        assert_eq!(regs[26], 3);
    }

    #[test]
    fn shifts_by_immediate_saturate() {
        let mut regs = zeroed();
        // sqshl v0.4s, v0.4s, #8
        regs[0] = 0x0100_0000 | (1u128 << 32);
        run(0x4f287400, &mut regs);
        assert_eq!(w32(regs[0], 0), 0x7fff_ffff);
        assert_eq!(w32(regs[0], 1), 0x100);
        // sqshlu v21.8h, v20.8h, #8: a negative lane is zero, and a large one saturates.
        regs[20] = 0x8000 | (0x0080u128 << 16) | (0x0100u128 << 32);
        run(0x6f186695, &mut regs);
        assert_eq!(lane(regs[21], 0, 2), 0);
        assert_eq!(lane(regs[21], 1, 2), 0x8000);
        assert_eq!(lane(regs[21], 2, 2), 0xffff);
    }

    #[test]
    fn shifts_by_register() {
        let mut regs = zeroed();
        // srshl v3.4s, v5.4s, v1.4s: a negative shift is a rounding right shift.
        regs[5] = 5;
        regs[1] = 0xffff_ffff; // -1
        run(0x4ea154a3, &mut regs);
        assert_eq!(w32(regs[3], 0), 3);
        // sqrshl v1.4s, v1.4s, v0.4s saturates left.
        regs[1] = 0x4000_0000;
        regs[0] = 2;
        run(0x4ea05c21, &mut regs);
        assert_eq!(w32(regs[1], 0), 0x7fff_ffff);
    }

    #[test]
    fn long_doubling_multiply_accumulate() {
        let mut regs = zeroed();
        // sqdmlal v2.4s, v3.4h, v0.4h
        regs[2] = 10;
        regs[3] = 3;
        regs[0] = 4;
        run(0x0e609062, &mut regs);
        assert_eq!(w32(regs[2], 0), 34);
        // sqdmlal2 v1.4s, v2.8h, v0.h[0] takes the upper half.
        regs[1] = 1;
        regs[2] = 7u128 << 64;
        regs[0] = 2;
        run(0x4f403041, &mut regs);
        assert_eq!(w32(regs[1], 0), 29);
    }

    #[test]
    fn polynomial_multiply() {
        let mut regs = zeroed();
        // pmull v0.1q, v0.1d, v1.1d: (x+1)(x+1) = x^2+1 over GF(2).
        regs[0] = 3;
        regs[1] = 3;
        run(0x0ee1e000, &mut regs);
        assert_eq!(regs[0], 5);
        // pmull2 takes the upper halves.
        regs[0] = 3u128 << 64;
        regs[1] = 5u128 << 64;
        run(0x4ee1e000, &mut regs);
        assert_eq!(regs[0], 15);
    }

    #[test]
    fn half_precision() {
        let mut regs = zeroed();
        // fcvt h0, s0
        regs[0] = 1.5f32.to_bits() as u128;
        run(0x1e23c000, &mut regs);
        assert_eq!(regs[0], 0x3e00);
        // 1 + 2^-11 is halfway between two halves: it rounds to even, 1.0.
        regs[0] = (1.0f32 + (2.0f32).powi(-11)).to_bits() as u128;
        run(0x1e23c000, &mut regs);
        assert_eq!(regs[0], 0x3c00);
        // fneg v2.8h, v0.8h
        regs[0] = 0x3c00;
        run(0x6ef8f802, &mut regs);
        assert_eq!(lane(regs[2], 0, 2), 0xbc00);
    }

    /// `fcvt h0, d0` rounds once: a double just below the halfway point between two halves
    /// rounds down, though single's nearest value to it *is* the halfway point.
    #[test]
    fn double_to_half_rounds_once() {
        let mut regs = zeroed();
        let fcvt_h0_d0 = 0x1e63c000;
        for (v, want) in [
            (1.0 + 2f64.powi(-11) - 2f64.powi(-40), 0x3c00), // below halfway: down
            (1.0 + 2f64.powi(-11) + 2f64.powi(-40), 0x3c01), // above: up
            (1.0 + 2f64.powi(-11), 0x3c00),                  // the tie: to even
            (-(1.0 + 3.0 * 2f64.powi(-11) - 2f64.powi(-40)), 0xbc01),
            (1.0e300, 0x7c00),
            (1.0e-300, 0x0000),
        ] {
            regs[0] = f64::to_bits(v) as u128;
            run(fcvt_h0_d0, &mut regs);
            assert_eq!(regs[0], want, "fcvt h, d of {v:e}");
        }
    }

    #[test]
    fn signed_add_long_pairwise_words() {
        let mut regs = zeroed();
        // saddlp v2.1d, v2.2s
        regs[2] = (-1i32 as u32 as u128) | (3u128 << 32);
        run(0x0ea02842, &mut regs);
        assert_eq!(regs[2], 2);
    }

    #[test]
    fn what_is_not_here_is_refused() {
        let mut bytes = [0u8; 512];
        let mut fpsr = 0;
        assert!(!execute(0, &mut VRegs(&mut bytes), &mut fpsr));
        assert!(!execute(0xd65f03c0, &mut VRegs(&mut bytes), &mut fpsr));
        assert_eq!(fpsr, 0);
    }
}
