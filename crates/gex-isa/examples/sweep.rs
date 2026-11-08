//! Sweep an A64 image for words [`gex_isa::decode`] cannot name: the
//! decoder's gaps against a real program's text, as `<addr> <raw>` lines to
//! cluster and classify offline (a gap shows up at run time as "illegal
//! instruction", one at a time; this finds them all at once).
//!
//!     cargo run -p gex-isa --example sweep -- <file> <base> <text-end>
//!
//! `base` is the guest address word 0 of the file sits at (hex or decimal),
//! `text-end` the file offset to stop at (the module's TEXT_END).

use std::env;
use std::fs;

fn main() {
    let a: Vec<String> = env::args().collect();
    let (Some(path), Some(base), Some(end)) = (a.get(1), a.get(2), a.get(3)) else {
        eprintln!("usage: sweep <file> <base> <text-end>");
        std::process::exit(2);
    };
    let parse = |s: &String| -> u64 {
        let s = s.trim_start_matches("0x").trim_start_matches("0X");
        u64::from_str_radix(s, 16).unwrap()
    };
    let base = parse(base);
    let end = parse(end) as usize;
    let bytes = fs::read(path).unwrap();
    let end = end.min(bytes.len()).saturating_sub(3) & !3;
    let known = |off: usize| -> bool {
        if off + 4 > bytes.len() {
            return false;
        }
        let raw = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        !matches!(
            gex_isa::decode(raw, base + off as u64),
            gex_isa::Inst::Unknown { .. } | gex_isa::Inst::Undefined { .. }
        )
    };
    let mut n = 0usize;
    for off in (0..end).step_by(4) {
        if known(off) {
            continue;
        }
        // In code: the words on both sides decode. A word inside a literal
        // pool, a jump table or a string is not reported -- those are data
        // the translator never fetches.
        if !off.checked_sub(4).is_some_and(known) || !known(off + 4) {
            continue;
        }
        let raw = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        println!("{:x} {:08x}", base + off as u64, raw);
        n += 1;
    }
    eprintln!("sweep: {} unknown words in code context, {} bytes", n, end);
}
