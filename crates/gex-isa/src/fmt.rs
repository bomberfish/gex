//! Disassembly text for [`Inst`].
//!
//! Output follows binutils syntax so it can be diffed against `objdump -d`.
//! Preferred aliases (`mov`, `cmp`, `lsl`, `ubfx`, …) are reproduced, since a
//! raw `orr`/`subs`/`ubfm` would show up as a spurious diff.

use crate::*;
use core::fmt::Write;

fn reg(r: RegSel, sz: Sz) -> String {
    let p = if sz.is64() { 'x' } else { 'w' };
    match r {
        RegSel::Gp(n) => format!("{p}{n}"),
        RegSel::Zr => format!("{p}zr"),
        RegSel::Sp => {
            if sz.is64() {
                "sp".into()
            } else {
                "wsp".into()
            }
        }
    }
}

fn shifted(rm: RegSel, sz: Sz, shift: ShiftType, amount: u32) -> String {
    if amount == 0 && shift == ShiftType::Lsl {
        reg(rm, sz)
    } else {
        format!("{}, {} #{amount}", reg(rm, sz), shift.name())
    }
}

fn extended(rm: RegSel, sz: Sz, ext: Extend, amount: u32) -> String {
    // The register printed is 32-bit for byte/half/word extensions.
    let rsz = if ext.src_bits() == 64 { Sz::X } else { Sz::W };
    let mut s = format!("{}, {}", reg(rm, rsz), ext.name());
    if amount != 0 {
        let _ = write!(s, " #{amount}");
    }
    // `uxtx`/`sxtx` with a 64-bit operand prints as `lsl` when it is the
    // default extension, matching binutils.
    if sz.is64() && ext == Extend::Uxtx {
        s = if amount == 0 {
            reg(rm, Sz::X)
        } else {
            format!("{}, lsl #{amount}", reg(rm, Sz::X))
        };
    }
    s
}

/// Data immediates print in hex, matching binutils.
fn imm(v: u64) -> String {
    format!("#0x{v:x}")
}

fn addr_str(a: &AddrMode) -> String {
    match *a {
        AddrMode::Offset { rn, off } => {
            if off == 0 {
                format!("[{}]", reg(rn, Sz::X))
            } else if off < 0 {
                format!("[{}, #-0x{:x}]", reg(rn, Sz::X), -off)
            } else {
                format!("[{}, #0x{:x}]", reg(rn, Sz::X), off)
            }
        }
        AddrMode::PostIndex { rn, off } => {
            if off < 0 {
                format!("[{}], #-0x{:x}", reg(rn, Sz::X), -off)
            } else {
                format!("[{}], #0x{:x}", reg(rn, Sz::X), off)
            }
        }
        AddrMode::PostIndexReg { rn, rm } => {
            format!("[{}], {}", reg(rn, Sz::X), reg(rm, Sz::X))
        }
        AddrMode::PreIndex { rn, off } => {
            if off < 0 {
                format!("[{}, #-0x{:x}]!", reg(rn, Sz::X), -off)
            } else {
                format!("[{}, #0x{:x}]!", reg(rn, Sz::X), off)
            }
        }
        AddrMode::RegOffset {
            rn,
            rm,
            ext,
            amount,
        } => {
            let rsz = if ext.src_bits() == 64 { Sz::X } else { Sz::W };
            let mut s = format!("[{}, {}", reg(rn, Sz::X), reg(rm, rsz));
            if ext == Extend::Uxtx {
                if amount != 0 {
                    let _ = write!(s, ", lsl #{amount}");
                }
            } else {
                let _ = write!(s, ", {}", ext.name());
                if amount != 0 {
                    let _ = write!(s, " #{amount}");
                }
            }
            s.push(']');
            s
        }
        AddrMode::Literal { addr } => format!("0x{addr:x}"),
    }
}

/// Render `inst` as assembly text. `pc` is unused for absolute operands (they
/// were resolved at decode time) but kept for symmetry with [`decode`].
pub fn disasm(inst: &Inst, _pc: u64) -> String {
    match *inst {
        Inst::Unknown { raw } => format!(".inst 0x{raw:08x} ; undecoded"),
        Inst::Undefined { raw } => format!("udf ; 0x{raw:08x}"),
        // Hint space. The PAC hints (`paciasp`, `autiasp`, …) are no-ops here:
        // sign and authenticate are both the identity without PAC.
        Inst::Hint { imm: h } => match h {
            0 => "nop".into(),
            1 => "yield".into(),
            2 => "wfe".into(),
            3 => "wfi".into(),
            4 => "sev".into(),
            5 => "sevl".into(),
            6 => "dgh".into(),
            7 => "xpaclri".into(),
            8 => "pacia1716".into(),
            10 => "pacib1716".into(),
            12 => "autia1716".into(),
            14 => "autib1716".into(),
            16 => "esb".into(),
            17 => "psb".into(),
            18 => "tsb".into(),
            20 => "csdb".into(),
            24 => "paciaz".into(),
            25 => "paciasp".into(),
            26 => "pacibz".into(),
            27 => "pacibsp".into(),
            28 => "autiaz".into(),
            29 => "autiasp".into(),
            30 => "autibz".into(),
            31 => "autibsp".into(),
            32 => "bti".into(),
            34 => "bti c".into(),
            36 => "bti j".into(),
            38 => "bti jc".into(),
            _ => format!("hint #{h}"),
        },
        Inst::Clrex => "clrex".into(),

        Inst::Adr { rd, target } => format!("adr {}, 0x{target:x}", reg(rd, Sz::X)),
        Inst::Adrp { rd, target } => format!("adrp {}, 0x{target:x}", reg(rd, Sz::X)),

        Inst::AddSubImm {
            sz,
            op,
            set_flags,
            rd,
            rn,
            imm: v,
        } => {
            let m = mnem_addsub(op, set_flags);
            // cmp/cmn alias: flags set and destination is the zero register.
            if set_flags && rd == RegSel::Zr {
                let alias = if op == AddOp::Sub { "cmp" } else { "cmn" };
                return format!("{alias} {}, {}", reg(rn, sz), imm(v));
            }
            // mov to/from sp: `add rd, rn, #0` with an SP operand.
            if v == 0 && !set_flags && op == AddOp::Add && (rd == RegSel::Sp || rn == RegSel::Sp) {
                return format!("mov {}, {}", reg(rd, sz), reg(rn, sz));
            }
            format!("{m} {}, {}, {}", reg(rd, sz), reg(rn, sz), imm(v))
        }
        Inst::LogicalImm {
            sz,
            op,
            set_flags,
            rd,
            rn,
            imm: v,
        } => {
            if set_flags && rd == RegSel::Zr {
                return format!("tst {}, {}", reg(rn, sz), imm(v));
            }
            if op == LogOp::Orr && rn == RegSel::Zr && !set_flags {
                return format!("mov {}, {}", reg(rd, sz), imm(v));
            }
            let m = match (op, set_flags) {
                (LogOp::And, false) => "and",
                (LogOp::And, true) => "ands",
                (LogOp::Orr, _) => "orr",
                (LogOp::Eor, _) => "eor",
            };
            format!("{m} {}, {}, {}", reg(rd, sz), reg(rn, sz), imm(v))
        }
        Inst::MoveWide {
            sz,
            op,
            rd,
            imm16,
            shift,
        } => {
            // `mov (wide immediate)` is the preferred alias for movz/movn
            // unless the immediate is zero with a non-zero shift.
            if op != MoveOp::Movk && !(imm16 == 0 && shift != 0) {
                let v = imm16 << shift;
                let v = if op == MoveOp::Movn { !v } else { v } & sz.mask();
                return format!("mov {}, #0x{v:x}", reg(rd, sz));
            }
            let m = match op {
                MoveOp::Movz => "movz",
                MoveOp::Movn => "movn",
                MoveOp::Movk => "movk",
            };
            if shift == 0 {
                format!("{m} {}, {}", reg(rd, sz), imm(imm16))
            } else {
                format!("{m} {}, {}, lsl #{shift}", reg(rd, sz), imm(imm16))
            }
        }
        Inst::Bitfield {
            sz,
            op,
            rd,
            rn,
            immr,
            imms,
        } => fmt_bitfield(sz, op, rd, rn, immr, imms),
        Inst::Extract {
            sz,
            rd,
            rn,
            rm,
            lsb,
        } => {
            if rn == rm {
                format!("ror {}, {}, #{lsb}", reg(rd, sz), reg(rn, sz))
            } else {
                format!(
                    "extr {}, {}, {}, #{lsb}",
                    reg(rd, sz),
                    reg(rn, sz),
                    reg(rm, sz)
                )
            }
        }

        Inst::AddSubShifted {
            sz,
            op,
            set_flags,
            rd,
            rn,
            rm,
            shift,
            amount,
        } => {
            if set_flags && rd == RegSel::Zr {
                let alias = if op == AddOp::Sub { "cmp" } else { "cmn" };
                return format!(
                    "{alias} {}, {}",
                    reg(rn, sz),
                    shifted(rm, sz, shift, amount)
                );
            }
            if op == AddOp::Sub && rn == RegSel::Zr {
                let alias = if set_flags { "negs" } else { "neg" };
                return format!(
                    "{alias} {}, {}",
                    reg(rd, sz),
                    shifted(rm, sz, shift, amount)
                );
            }
            format!(
                "{} {}, {}, {}",
                mnem_addsub(op, set_flags),
                reg(rd, sz),
                reg(rn, sz),
                shifted(rm, sz, shift, amount)
            )
        }
        Inst::AddSubExtended {
            sz,
            op,
            set_flags,
            rd,
            rn,
            rm,
            ext,
            amount,
        } => {
            if set_flags && rd == RegSel::Zr {
                let alias = if op == AddOp::Sub { "cmp" } else { "cmn" };
                return format!("{alias} {}, {}", reg(rn, sz), extended(rm, sz, ext, amount));
            }
            format!(
                "{} {}, {}, {}",
                mnem_addsub(op, set_flags),
                reg(rd, sz),
                reg(rn, sz),
                extended(rm, sz, ext, amount)
            )
        }
        Inst::AddSubCarry {
            sz,
            op,
            set_flags,
            rd,
            rn,
            rm,
        } => {
            // ngc/ngcs: sbc/sbcs with the zero register as first source.
            if op == AddOp::Sub && rn == RegSel::Zr {
                let alias = if set_flags { "ngcs" } else { "ngc" };
                return format!("{alias} {}, {}", reg(rd, sz), reg(rm, sz));
            }
            let m = match (op, set_flags) {
                (AddOp::Add, false) => "adc",
                (AddOp::Add, true) => "adcs",
                (AddOp::Sub, false) => "sbc",
                (AddOp::Sub, true) => "sbcs",
            };
            format!("{m} {}, {}, {}", reg(rd, sz), reg(rn, sz), reg(rm, sz))
        }
        Inst::LogicalShifted {
            sz,
            op,
            set_flags,
            invert,
            rd,
            rn,
            rm,
            shift,
            amount,
        } => {
            // mov (register) = orr rd, xzr, rm
            if op == LogOp::Orr
                && !invert
                && !set_flags
                && rn == RegSel::Zr
                && amount == 0
                && shift == ShiftType::Lsl
            {
                return format!("mov {}, {}", reg(rd, sz), reg(rm, sz));
            }
            // mvn = orn rd, xzr, rm
            if op == LogOp::Orr && invert && !set_flags && rn == RegSel::Zr {
                return format!("mvn {}, {}", reg(rd, sz), shifted(rm, sz, shift, amount));
            }
            if op == LogOp::And && set_flags && rd == RegSel::Zr && !invert {
                return format!("tst {}, {}", reg(rn, sz), shifted(rm, sz, shift, amount));
            }
            let m = match (op, invert, set_flags) {
                (LogOp::And, false, false) => "and",
                (LogOp::And, false, true) => "ands",
                (LogOp::And, true, false) => "bic",
                (LogOp::And, true, true) => "bics",
                (LogOp::Orr, false, _) => "orr",
                (LogOp::Orr, true, _) => "orn",
                (LogOp::Eor, false, _) => "eor",
                (LogOp::Eor, true, _) => "eon",
            };
            format!(
                "{m} {}, {}, {}",
                reg(rd, sz),
                reg(rn, sz),
                shifted(rm, sz, shift, amount)
            )
        }
        Inst::CondSelect {
            sz,
            rd,
            rn,
            rm,
            cond,
            else_inc,
            else_inv,
        } => {
            // cset/csetm: rn == rm == zr, condition inverted.
            if rn == RegSel::Zr && rm == RegSel::Zr && !cond.is_always() {
                if else_inc && !else_inv {
                    return format!("cset {}, {}", reg(rd, sz), cond.invert().name());
                }
                if else_inv && !else_inc {
                    return format!("csetm {}, {}", reg(rd, sz), cond.invert().name());
                }
            }
            // cinc/cinv/cneg: rn == rm, condition inverted.
            if rn == rm && rn != RegSel::Zr && !cond.is_always() {
                let alias = match (else_inv, else_inc) {
                    (false, true) => Some("cinc"),
                    (true, false) => Some("cinv"),
                    (true, true) => Some("cneg"),
                    (false, false) => None,
                };
                if let Some(alias) = alias {
                    return format!(
                        "{alias} {}, {}, {}",
                        reg(rd, sz),
                        reg(rn, sz),
                        cond.invert().name()
                    );
                }
            }
            let m = match (else_inv, else_inc) {
                (false, false) => "csel",
                (false, true) => "csinc",
                (true, false) => "csinv",
                (true, true) => "csneg",
            };
            format!(
                "{m} {}, {}, {}, {}",
                reg(rd, sz),
                reg(rn, sz),
                reg(rm, sz),
                cond.name()
            )
        }
        Inst::CondCompare {
            sz,
            op,
            rn,
            rm,
            imm: v,
            nzcv,
            cond,
        } => {
            let m = if op == AddOp::Sub { "ccmp" } else { "ccmn" };
            let second = match rm {
                Some(r) => reg(r, sz),
                None => format!("#{v}"),
            };
            format!("{m} {}, {second}, #{nzcv}, {}", reg(rn, sz), cond.name())
        }
        Inst::DataProc1 { sz, op, rd, rn } => {
            let m = match op {
                Dp1Op::Rbit => "rbit",
                Dp1Op::Rev16 => "rev16",
                Dp1Op::Rev32 => "rev32",
                Dp1Op::Rev => "rev",
                Dp1Op::Clz => "clz",
                Dp1Op::Cls => "cls",
            };
            format!("{m} {}, {}", reg(rd, sz), reg(rn, sz))
        }
        Inst::DataProc2 { sz, op, rd, rn, rm } => {
            let m = match op {
                Dp2Op::Udiv => "udiv",
                Dp2Op::Sdiv => "sdiv",
                Dp2Op::Lslv => "lsl",
                Dp2Op::Lsrv => "lsr",
                Dp2Op::Asrv => "asr",
                Dp2Op::Rorv => "ror",
                Dp2Op::Crc32b => "crc32b",
                Dp2Op::Crc32h => "crc32h",
                Dp2Op::Crc32w => "crc32w",
                Dp2Op::Crc32x => "crc32x",
                Dp2Op::Crc32cb => "crc32cb",
                Dp2Op::Crc32ch => "crc32ch",
                Dp2Op::Crc32cw => "crc32cw",
                Dp2Op::Crc32cx => "crc32cx",
            };
            format!("{m} {}, {}, {}", reg(rd, sz), reg(rn, sz), reg(rm, sz))
        }
        Inst::DataProc3 {
            sz,
            op,
            rd,
            rn,
            rm,
            ra,
        } => {
            let (m, wide) = match op {
                Dp3Op::Madd => ("madd", false),
                Dp3Op::Msub => ("msub", false),
                Dp3Op::Smaddl => ("smaddl", true),
                Dp3Op::Smsubl => ("smsubl", true),
                Dp3Op::Umaddl => ("umaddl", true),
                Dp3Op::Umsubl => ("umsubl", true),
                Dp3Op::Smulh => ("smulh", false),
                Dp3Op::Umulh => ("umulh", false),
            };
            let ssz = if wide { Sz::W } else { sz };
            if matches!(op, Dp3Op::Smulh | Dp3Op::Umulh) {
                return format!("{m} {}, {}, {}", reg(rd, sz), reg(rn, sz), reg(rm, sz));
            }
            if ra == RegSel::Zr {
                let alias = match op {
                    Dp3Op::Madd => "mul",
                    Dp3Op::Msub => "mneg",
                    Dp3Op::Smaddl => "smull",
                    Dp3Op::Smsubl => "smnegl",
                    Dp3Op::Umaddl => "umull",
                    Dp3Op::Umsubl => "umnegl",
                    _ => unreachable!(),
                };
                return format!(
                    "{alias} {}, {}, {}",
                    reg(rd, sz),
                    reg(rn, ssz),
                    reg(rm, ssz)
                );
            }
            format!(
                "{m} {}, {}, {}, {}",
                reg(rd, sz),
                reg(rn, ssz),
                reg(rm, ssz),
                reg(ra, sz)
            )
        }

        Inst::Branch { target, link } => {
            format!("{} 0x{target:x}", if link { "bl" } else { "b" })
        }
        Inst::BranchCond { cond, target } => format!("b.{} 0x{target:x}", cond.name()),
        Inst::CompareBranch {
            sz,
            rt,
            target,
            nonzero,
        } => format!(
            "{} {}, 0x{target:x}",
            if nonzero { "cbnz" } else { "cbz" },
            reg(rt, sz)
        ),
        Inst::TestBranch {
            rt,
            bit,
            target,
            nonzero,
        } => format!(
            "{} {}, #{bit}, 0x{target:x}",
            if nonzero { "tbnz" } else { "tbz" },
            reg(rt, if bit >= 32 { Sz::X } else { Sz::W })
        ),
        Inst::BranchReg { rn, link, is_ret } => {
            if is_ret {
                if rn == RegSel::Gp(30) {
                    "ret".into()
                } else {
                    format!("ret {}", reg(rn, Sz::X))
                }
            } else if link {
                format!("blr {}", reg(rn, Sz::X))
            } else {
                format!("br {}", reg(rn, Sz::X))
            }
        }

        Inst::Svc { imm: v } => format!("svc #0x{v:x}"),
        Inst::Brk { imm: v } => format!("brk #0x{v:x}"),

        Inst::SysRegRead { rt, sysreg } => {
            format!("mrs {}, {}", reg(rt, Sz::X), sysreg_name(sysreg))
        }
        Inst::SysRegWrite { rt, sysreg } => {
            format!("msr {}, {}", sysreg_name(sysreg), reg(rt, Sz::X))
        }
        Inst::SysMaintenance { op, rt } => {
            let m = match op {
                SysMaintOp::DcZva => "dc zva",
                SysMaintOp::DcClean => "dc",
                SysMaintOp::IcInvalidate => "ic",
                SysMaintOp::Other => "sys",
            };
            format!("{m} {}", reg(rt, Sz::X))
        }
        Inst::Barrier { kind, option } => {
            let m = match kind {
                BarrierKind::Dmb => "dmb",
                BarrierKind::Dsb => "dsb",
                BarrierKind::Isb => "isb",
                BarrierKind::Sb => "sb",
            };
            if kind == BarrierKind::Isb {
                return "isb".into();
            }
            let o = match option {
                0b1111 => "sy",
                0b1110 => "st",
                0b1101 => "ld",
                0b1011 => "ish",
                0b1010 => "ishst",
                0b1001 => "ishld",
                0b0111 => "nsh",
                0b0110 => "nshst",
                0b0011 => "osh",
                _ => "sy",
            };
            format!("{m} {o}")
        }

        Inst::Load {
            sz,
            dest,
            signed,
            rt,
            ref addr,
            order,
        } => {
            let m = load_mnemonic(sz, dest, signed, order, addr);
            format!("{m} {}, {}", reg(rt, dest), addr_str(addr))
        }
        Inst::Store {
            sz,
            rt,
            ref addr,
            order,
        } => {
            let base = if order == Ordering::Release {
                match sz {
                    MemSz::B => "stlrb",
                    MemSz::H => "stlrh",
                    _ => "stlr",
                }
            } else {
                match sz {
                    MemSz::B => "strb",
                    MemSz::H => "strh",
                    _ => "str",
                }
            };
            let rsz = if sz == MemSz::D { Sz::X } else { Sz::W };
            format!("{base} {}, {}", reg(rt, rsz), addr_str(addr))
        }
        Inst::LoadPair {
            sz,
            signed,
            rt,
            rt2,
            ref addr,
        } => {
            let rsz = if sz == MemSz::D || signed {
                Sz::X
            } else {
                Sz::W
            };
            let m = if signed { "ldpsw" } else { "ldp" };
            format!(
                "{m} {}, {}, {}",
                reg(rt, rsz),
                reg(rt2, rsz),
                addr_str(addr)
            )
        }
        Inst::StorePair {
            sz,
            rt,
            rt2,
            ref addr,
        } => {
            let rsz = if sz == MemSz::D { Sz::X } else { Sz::W };
            format!(
                "stp {}, {}, {}",
                reg(rt, rsz),
                reg(rt2, rsz),
                addr_str(addr)
            )
        }
        Inst::TagStore { ref addr, zero } => {
            let m = match zero {
                16 => "stzg",
                32 => "stz2g",
                _ => "stg",
            };
            format!("{m} {}", addr_str(addr))
        }
        Inst::Prefetch { ref addr, hint } => {
            // `Rt` is really three fields: what to prefetch, into which cache
            // level, and whether to keep it there.
            let kind = ["pld", "pli", "pst", "#"][(hint >> 3) as usize & 3];
            let name = if kind == "#" {
                format!("#0x{hint:x}")
            } else {
                format!(
                    "{kind}l{}{}",
                    ((hint >> 1) & 3) + 1,
                    if hint & 1 == 1 { "strm" } else { "keep" }
                )
            };
            format!("prfm {name}, {}", addr_str(addr))
        }

        Inst::VLoad {
            bytes,
            vt,
            ref addr,
        } => format!("ldr {}, {}", vreg_scalar(vt, bytes), addr_str(addr)),
        Inst::VStore {
            bytes,
            vt,
            ref addr,
        } => format!("str {}, {}", vreg_scalar(vt, bytes), addr_str(addr)),
        Inst::VLoadPair {
            bytes,
            vt,
            vt2,
            ref addr,
        } => format!(
            "ldp {}, {}, {}",
            vreg_scalar(vt, bytes),
            vreg_scalar(vt2, bytes),
            addr_str(addr)
        ),
        Inst::VStorePair {
            bytes,
            vt,
            vt2,
            ref addr,
        } => format!(
            "stp {}, {}, {}",
            vreg_scalar(vt, bytes),
            vreg_scalar(vt2, bytes),
            addr_str(addr)
        ),
        Inst::VDupGeneral {
            vd,
            rn,
            lane_bytes,
            q,
        } => format!(
            "dup {}, {}",
            vreg_vector(vd, lane_bytes, q),
            reg(rn, if lane_bytes == 8 { Sz::X } else { Sz::W })
        ),
        // The immediate is already expanded, so the lane shape the assembler
        // chose is not recoverable; print the resulting register value.
        Inst::VMoveImm {
            vd,
            value,
            inverted,
            fp_lane_bytes,
            cmode,
            imm8,
            q,
            op,
        } => match fp_lane_bytes {
            // The floating-point form does know its arrangement, because
            // `cmode` fixes the lane width and Q the count.
            Some(lb) => {
                let lanes = (if q { 16 } else { 8 }) / lb;
                let v = if lb == 8 {
                    f64::from_bits(value as u64)
                } else {
                    f32::from_bits(value as u32) as f64
                };
                format!(
                    "fmov v{}.{lanes}{}, #{}",
                    vd.0,
                    lane_letter(lb),
                    fp_imm_text(v)
                )
            }
            None => {
                let m = if inverted { "mvni" } else { "movi" };
                let (lane_bytes, shift, msl) = match (cmode >> 1, cmode & 1) {
                    (c, 0) if c < 4 => (4, 8 * c as u32, false),
                    (c, 0) if (4..6).contains(&c) => (2, 8 * (c as u32 - 4), false),
                    (6, _) => (4, 8 * (1 + (cmode as u32 & 1)), true),
                    // `cmode` 1110 with op set is the only 64-bit lane form,
                    // and the only one whose immediate is worth printing whole.
                    (7, _) if op => {
                        return if q {
                            format!("movi v{}.2d, #0x{:x}", vd.0, value as u64)
                        } else {
                            format!("movi d{}, #0x{:x}", vd.0, value as u64)
                        };
                    }
                    _ => (1, 0, false),
                };
                let arr = VArr {
                    lane_bytes,
                    q,
                    float: false,
                    scalar: false,
                };
                let tail = match (shift, msl) {
                    (0, false) => String::new(),
                    (s, false) => format!(", lsl #{s}"),
                    (s, true) => format!(", msl #{s}"),
                };
                format!("{m} v{}.{}, #0x{imm8:x}{tail}", vd.0, arr.suffix())
            }
        },
        Inst::FpDataProc1 { sz, op, vd, vn } => {
            let m = match op {
                FpOp1::Mov => "fmov",
                FpOp1::Abs => "fabs",
                FpOp1::Neg => "fneg",
                FpOp1::Sqrt => "fsqrt",
                FpOp1::RoundNearest => "frintn",
                FpOp1::RoundPlus => "frintp",
                FpOp1::RoundMinus => "frintm",
                FpOp1::RoundZero => "frintz",
                FpOp1::RoundAway => "frinta",
                FpOp1::RoundExact => "frintx",
                FpOp1::RoundCurrent => "frinti",
            };
            format!("{m} {}, {}", freg(vd, sz), freg(vn, sz))
        }
        Inst::FpDataProc2 { sz, op, vd, vn, vm } => {
            let m = match op {
                FpOp2::Mul => "fmul",
                FpOp2::Div => "fdiv",
                FpOp2::Add => "fadd",
                FpOp2::Sub => "fsub",
                FpOp2::Max => "fmax",
                FpOp2::Min => "fmin",
                FpOp2::MaxNum => "fmaxnm",
                FpOp2::MinNum => "fminnm",
                FpOp2::NMul => "fnmul",
            };
            format!("{m} {}, {}, {}", freg(vd, sz), freg(vn, sz), freg(vm, sz))
        }
        Inst::FpDataProc3 {
            sz,
            vd,
            vn,
            vm,
            va,
            neg_product,
            neg_addend,
        } => {
            let m = match (neg_product, neg_addend) {
                (false, false) => "fmadd",
                (true, false) => "fmsub",
                (true, true) => "fnmadd",
                (false, true) => "fnmsub",
            };
            format!(
                "{m} {}, {}, {}, {}",
                freg(vd, sz),
                freg(vn, sz),
                freg(vm, sz),
                freg(va, sz)
            )
        }
        Inst::FpCompare { sz, vn, vm } => match vm {
            Some(vm) => format!("fcmp {}, {}", freg(vn, sz), freg(vm, sz)),
            None => format!("fcmp {}, #0.0", freg(vn, sz)),
        },
        Inst::FpCondSelect {
            sz,
            vd,
            vn,
            vm,
            cond,
        } => format!(
            "fcsel {}, {}, {}, {}",
            freg(vd, sz),
            freg(vn, sz),
            freg(vm, sz),
            cond.name()
        ),
        Inst::FpMoveImm { sz, vd, bits } => {
            let value = if sz.is_double() {
                f64::from_bits(bits)
            } else {
                f32::from_bits(bits as u32) as f64
            };
            format!("fmov {}, #{}", freg(vd, sz), fp_imm_text(value))
        }
        Inst::FpConvertPrecision { from, to, vd, vn } => {
            format!("fcvt {}, {}", freg(vd, to), freg(vn, from))
        }
        Inst::FpFromInt {
            sz,
            int_sz,
            signed,
            vd,
            rn,
            fbits,
        } => format!(
            "{} {}, {}{}",
            if signed { "scvtf" } else { "ucvtf" },
            freg(vd, sz),
            reg(rn, int_sz),
            fbits_suffix(fbits)
        ),
        Inst::FpToInt {
            sz,
            int_sz,
            signed,
            round,
            rd,
            vn,
            fbits,
        } => {
            let r = match round {
                FpRound::Nearest => 'n',
                FpRound::Plus => 'p',
                FpRound::Minus => 'm',
                FpRound::Zero => 'z',
                FpRound::Away => 'a',
            };
            format!(
                "fcvt{r}{} {}, {}{}",
                if signed { "s" } else { "u" },
                reg(rd, int_sz),
                freg(vn, sz),
                fbits_suffix(fbits)
            )
        }
        Inst::FpMoveGeneral {
            sz,
            to_vector,
            rd,
            vd,
        } => {
            let int_sz = if sz.is_double() { Sz::X } else { Sz::W };
            if to_vector {
                format!("fmov {}, {}", freg(vd, sz), reg(rd, int_sz))
            } else {
                format!("fmov {}, {}", reg(rd, int_sz), freg(vd, sz))
            }
        }

        Inst::FpMoveTopHalf { to_vector, rd, vd } => {
            if to_vector {
                format!("fmov v{}.d[1], {}", vd.0, reg(rd, Sz::X))
            } else {
                format!("fmov {}, v{}.d[1]", reg(rd, Sz::X), vd.0)
            }
        }

        Inst::VMoveLane {
            to_vector,
            signed,
            lane_bytes,
            lane,
            int_sz,
            rd,
            vd,
        } => {
            let t = match lane_bytes {
                1 => 'b',
                2 => 'h',
                4 => 's',
                _ => 'd',
            };
            if to_vector {
                format!("mov v{}.{t}[{lane}], {}", vd.0, reg(rd, int_sz))
            } else if signed {
                format!("smov {}, v{}.{t}[{lane}]", reg(rd, int_sz), vd.0)
            } else {
                // binutils prints the full-width extraction as `mov`.
                let m = if lane_bytes == 8 || (lane_bytes == 4 && int_sz == Sz::W) {
                    "mov"
                } else {
                    "umov"
                };
                format!("{m} {}, v{}.{t}[{lane}]", reg(rd, int_sz), vd.0)
            }
        }

        Inst::VLoadMulti {
            regs,
            vt,
            bytes,
            ref addr,
        }
        | Inst::VStoreMulti {
            regs,
            vt,
            bytes,
            ref addr,
        } => {
            let m = if matches!(*inst, Inst::VLoadMulti { .. }) {
                "ld1"
            } else {
                "st1"
            };
            let w = if bytes == 16 { "16b" } else { "8b" };
            let list: Vec<String> = (0..regs)
                .map(|i| format!("v{}.{w}", (vt.0 + i) % 32))
                .collect();
            format!("{m} {{{}}}, {}", list.join(", "), addr_str(addr))
        }
        Inst::VLoadStoreLane {
            load,
            replicate,
            regs,
            lane_bytes,
            index,
            q,
            vt,
            ref addr,
        } => {
            let l = lane_letter(lane_bytes);
            let list = |suffix: &str| -> String {
                (0..regs as u32)
                    .map(|i| format!("v{}.{suffix}", (vt.0 as u32 + i) % 32))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            if replicate {
                let lanes = (if q { 16 } else { 8 }) / lane_bytes;
                let m = ["ld1r", "ld2r", "ld3r", "ld4r"][(regs - 1) as usize];
                format!("{m} {{{}}}, {}", list(&format!("{lanes}{l}")), addr_str(addr))
            } else {
                let base = if load { "ld" } else { "st" };
                format!(
                    "{base}{regs} {{{}}}[{index}], {}",
                    list(&l.to_string()),
                    addr_str(addr)
                )
            }
        }
        Inst::VLoadStore2 {
            load,
            lane_bytes,
            bytes,
            vt,
            ref addr,
        } => {
            let arr = VArr {
                lane_bytes,
                q: bytes == 16,
                float: false,
                scalar: false,
            };
            format!(
                "{} {{v{}.{}, v{}.{}}}, {}",
                if load { "ld2" } else { "st2" },
                vt.0,
                arr.suffix(),
                (vt.0 + 1) % 32,
                arr.suffix(),
                addr_str(addr)
            )
        }
        Inst::VLoadStoreN {
            load,
            regs,
            lane_bytes,
            bytes,
            vt,
            ref addr,
        } => {
            let arr = VArr {
                lane_bytes,
                q: bytes == 16,
                float: false,
                scalar: false,
            };
            let list: Vec<String> = (0..regs)
                .map(|i| format!("v{}.{}", (vt.0 + i) % 32, arr.suffix()))
                .collect();
            format!(
                "{}{regs} {{{}}}, {}",
                if load { "ld" } else { "st" },
                list.join(", "),
                addr_str(addr)
            )
        }
        Inst::VCnt { vd, vn, q } => {
            let w = if q { "16b" } else { "8b" };
            format!("cnt v{}.{w}, v{}.{w}", vd.0, vn.0)
        }
        Inst::VAcross { op, arr, vd, vn } => {
            // The scalar forms reduce one pair, so they are spelled `-p`
            // rather than `-v`, and their source is still a two-lane vector.
            let m = match (op, arr.scalar) {
                (VAcrossOp::Add, false) => "addv",
                (VAcrossOp::SAddLv, _) => "saddlv",
                (VAcrossOp::UAddLv, _) => "uaddlv",
                (VAcrossOp::Add, true) if arr.float => "faddp",
                (VAcrossOp::Add, true) => "addp",
                (VAcrossOp::SMax, _) => "smaxv",
                (VAcrossOp::SMin, _) => "sminv",
                (VAcrossOp::UMax, _) => "umaxv",
                (VAcrossOp::UMin, _) => "uminv",
                (VAcrossOp::FMax, false) => "fmaxv",
                (VAcrossOp::FMax, true) => "fmaxp",
                (VAcrossOp::FMin, false) => "fminv",
                (VAcrossOp::FMin, true) => "fminp",
                (VAcrossOp::FMaxNm, false) => "fmaxnmv",
                (VAcrossOp::FMaxNm, true) => "fmaxnmp",
                (VAcrossOp::FMinNm, false) => "fminnmv",
                (VAcrossOp::FMinNm, true) => "fminnmp",
            };
            let src = VArr {
                scalar: false,
                ..arr
            };
            format!(
                "{m} {}{}, v{}.{}",
                lane_letter(arr.lane_bytes),
                vd.0,
                vn.0,
                src.suffix()
            )
        }

        Inst::VSelect {
            kind,
            vd,
            vn,
            vm,
            q,
        } => {
            let w = if q { "16b" } else { "8b" };
            let m = match kind {
                VSelKind::Bsl => "bsl",
                VSelKind::Bit => "bit",
                VSelKind::Bif => "bif",
            };
            format!("{m} v{}.{w}, v{}.{w}, v{}.{w}", vd.0, vn.0, vm.0)
        }
        Inst::VThreeSame {
            op,
            arr,
            vd,
            vn,
            vm,
        } => {
            let m = vop3_name(op);
            format!(
                "{m} {}, {}, {}",
                areg(vd, arr),
                areg(vn, arr),
                areg(vm, arr)
            )
        }
        Inst::VTwoMisc { op, arr, vd, vn } => {
            let m = vop2_name(op);
            let a = arr.suffix();
            // The compare-with-zero forms print the zero operand.
            if matches!(
                op,
                VOp2::CmEqZero | VOp2::CmGtZero | VOp2::CmGeZero | VOp2::CmLtZero | VOp2::CmLeZero
            ) {
                return format!("{m} {}, {}, #0", areg(vd, arr), areg(vn, arr));
            }
            if matches!(
                op,
                VOp2::FCmEqZero
                    | VOp2::FCmGtZero
                    | VOp2::FCmGeZero
                    | VOp2::FCmLtZero
                    | VOp2::FCmLeZero
            ) {
                return format!("{m} {}, {}, #0.0", areg(vd, arr), areg(vn, arr));
            }
            if op == VOp2::Shll {
                let wide = VArr {
                    lane_bytes: arr.lane_bytes * 2,
                    q: true,
                    float: false,
                    scalar: false,
                };
                return format!(
                    "{m}{} v{}.{}, {}, #{}",
                    if arr.q { "2" } else { "" },
                    vd.0,
                    wide.suffix(),
                    areg(vn, arr),
                    arr.lane_bytes * 8
                );
            }
            // The narrowing forms name the wider source arrangement.
            if matches!(op, VOp2::Xtn | VOp2::SqXtn | VOp2::UqXtn | VOp2::SqXtun) {
                if arr.scalar {
                    return format!(
                        "{m} {}{}, {}{}",
                        lane_letter(arr.lane_bytes),
                        vd.0,
                        lane_letter(arr.lane_bytes * 2),
                        vn.0
                    );
                }
                let wide = VArr {
                    lane_bytes: arr.lane_bytes * 2,
                    q: true,
                    float: false,
                    scalar: false,
                };
                return format!(
                    "{m}{} v{}.{a}, v{}.{}",
                    if arr.q { "2" } else { "" },
                    vd.0,
                    vn.0,
                    wide.suffix()
                );
            }
            format!("{m} {}, {}", areg(vd, arr), areg(vn, arr))
        }
        Inst::VPermute {
            op,
            arr,
            vd,
            vn,
            vm,
        } => {
            let m = match op {
                VPermOp::Trn1 => "trn1",
                VPermOp::Trn2 => "trn2",
                VPermOp::Uzp1 => "uzp1",
                VPermOp::Uzp2 => "uzp2",
                VPermOp::Zip1 => "zip1",
                VPermOp::Zip2 => "zip2",
            };
            format!(
                "{m} {}, {}, {}",
                areg(vd, arr),
                areg(vn, arr),
                areg(vm, arr)
            )
        }
        Inst::VExt {
            vd,
            vn,
            vm,
            index,
            q,
        } => {
            let w = if q { "16b" } else { "8b" };
            format!("ext v{}.{w}, v{}.{w}, v{}.{w}, #{index}", vd.0, vn.0, vm.0)
        }
        Inst::VTableLookup {
            vd,
            vn,
            regs,
            vm,
            q,
            is_tbx,
        } => {
            let w = if q { "16b" } else { "8b" };
            let list: Vec<String> = (0..regs)
                .map(|i| format!("v{}.16b", (vn.0 + i) % 32))
                .collect();
            format!(
                "{} v{}.{w}, {{{}}}, v{}.{w}",
                if is_tbx { "tbx" } else { "tbl" },
                vd.0,
                list.join(", "),
                vm.0
            )
        }
        Inst::VDupElement {
            vd,
            vn,
            lane_bytes,
            index,
            q: _,
            scalar: true,
        } => format!(
            "mov {}{}, v{}.{}[{index}]",
            lane_letter(lane_bytes),
            vd.0,
            vn.0,
            lane_letter(lane_bytes)
        ),
        Inst::VDupElement {
            vd,
            vn,
            lane_bytes,
            index,
            q,
            scalar: false,
        } => {
            let arr = VArr {
                lane_bytes,
                q,
                float: false,
                scalar: false,
            };
            format!(
                "dup v{}.{}, v{}.{}[{index}]",
                vd.0,
                arr.suffix(),
                vn.0,
                lane_letter(lane_bytes)
            )
        }
        Inst::VInsElement {
            vd,
            vn,
            lane_bytes,
            dst_index,
            src_index,
        } => {
            let t = lane_letter(lane_bytes);
            format!("mov v{}.{t}[{dst_index}], v{}.{t}[{src_index}]", vd.0, vn.0)
        }
        Inst::VShiftImm {
            op,
            arr,
            vd,
            vn,
            shift,
        } => {
            let m = match op {
                VShiftImmOp::Shl => "shl",
                VShiftImmOp::SShr => "sshr",
                VShiftImmOp::UShr => "ushr",
                VShiftImmOp::SShll => "sshll",
                VShiftImmOp::UShll => "ushll",
                VShiftImmOp::SSra => "ssra",
                VShiftImmOp::USra => "usra",
                VShiftImmOp::Srshr => "srshr",
                VShiftImmOp::Urshr => "urshr",
                VShiftImmOp::Srsra => "srsra",
                VShiftImmOp::Ursra => "ursra",
                VShiftImmOp::Sli => "sli",
                VShiftImmOp::Sri => "sri",
                VShiftImmOp::SqShl => "sqshl",
                VShiftImmOp::UqShl => "uqshl",
                VShiftImmOp::SqShlU => "sqshlu",
                VShiftImmOp::Shrn => "shrn",
                VShiftImmOp::Rshrn => "rshrn",
                VShiftImmOp::SqShrn => "sqshrn",
                VShiftImmOp::SqRshrn => "sqrshrn",
                VShiftImmOp::UqShrn => "uqshrn",
                VShiftImmOp::UqRshrn => "uqrshrn",
                VShiftImmOp::SqShrun => "sqshrun",
                VShiftImmOp::SqRshrun => "sqrshrun",
                VShiftImmOp::Scvtf => "scvtf",
                VShiftImmOp::Ucvtf => "ucvtf",
                VShiftImmOp::Fcvtzs => "fcvtzs",
                VShiftImmOp::Fcvtzu => "fcvtzu",
            };
            // The width-changing forms name two arrangements: the wide one
            // is always a full 128 bits, and `sxtl`/`uxtl` are the aliases
            // for a zero shift.
            let wide = VArr {
                lane_bytes: arr.lane_bytes * 2,
                q: true,
                float: false,
                scalar: false,
            };
            match op {
                VShiftImmOp::SShll | VShiftImmOp::UShll => {
                    let signed = op == VShiftImmOp::SShll;
                    if shift == 0 {
                        format!(
                            "{}xtl{} v{}.{}, v{}.{}",
                            if signed { 's' } else { 'u' },
                            if arr.q { "2" } else { "" },
                            vd.0,
                            wide.suffix(),
                            vn.0,
                            arr.suffix()
                        )
                    } else {
                        format!(
                            "{m}{} v{}.{}, v{}.{}, #{shift}",
                            if arr.q { "2" } else { "" },
                            vd.0,
                            wide.suffix(),
                            vn.0,
                            arr.suffix()
                        )
                    }
                }
                VShiftImmOp::Shrn
                | VShiftImmOp::Rshrn
                | VShiftImmOp::SqShrn
                | VShiftImmOp::SqRshrn
                | VShiftImmOp::UqShrn
                | VShiftImmOp::UqRshrn
                | VShiftImmOp::SqShrun
                | VShiftImmOp::SqRshrun
                    if !arr.scalar =>
                {
                    format!(
                        "{m}{} v{}.{}, v{}.{}, #{shift}",
                        if arr.q { "2" } else { "" },
                        vd.0,
                        arr.suffix(),
                        vn.0,
                        wide.suffix()
                    )
                }
                // The scalar narrowing forms: `sqshrn s0, d1, #3`.
                VShiftImmOp::SqShrn
                | VShiftImmOp::SqRshrn
                | VShiftImmOp::UqShrn
                | VShiftImmOp::UqRshrn
                | VShiftImmOp::SqShrun
                | VShiftImmOp::SqRshrun => {
                    let wide = VArr { scalar: true, ..wide };
                    format!("{m} {}, {}, #{shift}", areg(vd, arr), areg(vn, wide))
                }
                _ => {
                    format!("{m} {}, {}, #{shift}", areg(vd, arr), areg(vn, arr))
                }
            }
        }
        Inst::VByElement {
            op,
            arr,
            vd,
            vn,
            vm,
            index,
        } => format!(
            "{} v{}.{}, v{}.{}, v{}.{}[{index}]",
            vop3_name(op),
            vd.0,
            arr.suffix(),
            vn.0,
            arr.suffix(),
            vm.0,
            arr.letter()
        ),

        Inst::VWiden {
            op,
            arr,
            upper,
            signed,
            vd,
            vn,
            vm,
            index,
        } => {
            let m = match op {
                VWidenOp::AddL => "addl",
                VWidenOp::AddW => "addw",
                VWidenOp::SubL => "subl",
                VWidenOp::SubW => "subw",
                VWidenOp::MulL => "mull",
                VWidenOp::MlaL => "mlal",
                VWidenOp::MlsL => "mlsl",
                VWidenOp::AbdL => "abdl",
                VWidenOp::AbaL => "abal",
            };
            let wide = VArr {
                lane_bytes: arr.lane_bytes * 2,
                q: true,
                float: false,
                scalar: false,
            };
            // The `w` forms take one wide operand and one narrow one.
            let n_arr = if matches!(op, VWidenOp::AddW | VWidenOp::SubW) {
                wide
            } else {
                arr
            };
            // The by-element form names one lane of `vm` (`v2.h[3]`) rather than
            // the whole vector.
            let m_operand = match index {
                Some(i) => format!("v{}.{}[{}]", vm.0, arr.letter(), i),
                None => format!("v{}.{}", vm.0, arr.suffix()),
            };
            format!(
                "{}{m}{} v{}.{}, v{}.{}, {m_operand}",
                if signed { 's' } else { 'u' },
                if upper { "2" } else { "" },
                vd.0,
                wide.suffix(),
                vn.0,
                n_arr.suffix(),
            )
        }

        Inst::VLogicalImm {
            vd,
            value,
            clear,
            q,
        } => {
            let arr = VArr {
                lane_bytes: 4,
                q,
                float: false,
                scalar: false,
            };
            format!(
                "{} v{}.{}, #{:#x}",
                if clear { "bic" } else { "orr" },
                vd.0,
                arr.suffix(),
                value as u32
            )
        }

        Inst::VNarrow {
            op,
            dst_bytes,
            upper,
            vd,
            vn,
            vm,
        } => {
            let m = match op {
                VNarrowOp::AddHn => "addhn",
                VNarrowOp::RAddHn => "raddhn",
                VNarrowOp::SubHn => "subhn",
                VNarrowOp::RSubHn => "rsubhn",
            };
            let dst = VArr {
                lane_bytes: dst_bytes,
                q: upper,
                float: false,
                scalar: false,
            };
            let src = VArr {
                lane_bytes: dst_bytes * 2,
                q: true,
                float: false,
                scalar: false,
            };
            format!(
                "{m}{} v{}.{}, v{}.{}, v{}.{}",
                if upper { "2" } else { "" },
                vd.0,
                dst.suffix(),
                vn.0,
                src.suffix(),
                vm.0,
                src.suffix()
            )
        }

        Inst::VFpCvtPrecision {
            widen,
            narrow_bytes,
            upper,
            vd,
            vn,
        } => {
            let narrow = VArr {
                lane_bytes: narrow_bytes,
                q: upper,
                float: true,
                scalar: false,
            };
            let wide = VArr {
                lane_bytes: narrow_bytes * 2,
                q: true,
                float: true,
                scalar: false,
            };
            let (d, s) = if widen {
                (wide, narrow)
            } else {
                (narrow, wide)
            };
            format!(
                "fcvt{}{} v{}.{}, v{}.{}",
                if widen { 'l' } else { 'n' },
                if upper { "2" } else { "" },
                vd.0,
                d.suffix(),
                vn.0,
                s.suffix()
            )
        }

        Inst::FpCondCompare {
            sz,
            vn,
            vm,
            nzcv,
            cond,
            signalling,
        } => format!(
            "fccmp{} {}, {}, #{nzcv}, {}",
            if signalling { "e" } else { "" },
            freg(vn, sz),
            freg(vm, sz),
            cond.name()
        ),

        Inst::VLogical { op, vd, vn, vm, q } => {
            let w = if q { "16b" } else { "8b" };
            if op == VLogOp::Orr && vn == vm {
                return format!("mov v{}.{w}, v{}.{w}", vd.0, vn.0);
            }
            let m = match op {
                VLogOp::And => "and",
                VLogOp::Bic => "bic",
                VLogOp::Orr => "orr",
                VLogOp::Orn => "orn",
                VLogOp::Eor => "eor",
            };
            format!("{m} v{}.{w}, v{}.{w}, v{}.{w}", vd.0, vn.0, vm.0)
        }

        Inst::LoadExclusive { sz, rt, rn, order } => {
            let acq = order == Ordering::Acquire;
            let m = match (sz, acq) {
                (MemSz::B, false) => "ldxrb",
                (MemSz::B, true) => "ldaxrb",
                (MemSz::H, false) => "ldxrh",
                (MemSz::H, true) => "ldaxrh",
                (_, false) => "ldxr",
                (_, true) => "ldaxr",
            };
            let rsz = if sz == MemSz::D { Sz::X } else { Sz::W };
            format!("{m} {}, [{}]", reg(rt, rsz), reg(rn, Sz::X))
        }
        Inst::StoreExclusive {
            sz,
            rs,
            rt,
            rn,
            order,
        } => {
            let rel = order == Ordering::Release;
            let m = match (sz, rel) {
                (MemSz::B, false) => "stxrb",
                (MemSz::B, true) => "stlxrb",
                (MemSz::H, false) => "stxrh",
                (MemSz::H, true) => "stlxrh",
                (_, false) => "stxr",
                (_, true) => "stlxr",
            };
            let rsz = if sz == MemSz::D { Sz::X } else { Sz::W };
            format!(
                "{m} {}, {}, [{}]",
                reg(rs, Sz::W),
                reg(rt, rsz),
                reg(rn, Sz::X)
            )
        }
        Inst::CompareSwap {
            sz,
            rs,
            rt,
            rn,
            acquire,
            release,
        } => {
            let suffix = match (acquire, release) {
                (false, false) => "",
                (true, false) => "a",
                (false, true) => "l",
                (true, true) => "al",
            };
            let w = match sz {
                MemSz::B => "b",
                MemSz::H => "h",
                _ => "",
            };
            let rsz = if sz == MemSz::D { Sz::X } else { Sz::W };
            format!(
                "cas{suffix}{w} {}, {}, [{}]",
                reg(rs, rsz),
                reg(rt, rsz),
                reg(rn, Sz::X)
            )
        }
        Inst::CompareSwapPair {
            sz,
            rs,
            rt,
            rn,
            acquire,
            release,
        } => {
            let suffix = match (acquire, release) {
                (false, false) => "",
                (true, false) => "a",
                (false, true) => "l",
                (true, true) => "al",
            };
            let rsz = if sz == MemSz::D { Sz::X } else { Sz::W };
            // The second register of each pair is the base + 1 (the
            // encoding keeps both bases even).
            let next = |r: RegSel| match r {
                RegSel::Gp(n) => RegSel::gp(n as u32 + 1),
                other => other,
            };
            format!(
                "casp{suffix} {}, {}, {}, {}, [{}]",
                reg(rs, rsz),
                reg(next(rs), rsz),
                reg(rt, rsz),
                reg(next(rt), rsz),
                reg(rn, Sz::X)
            )
        }
        Inst::AtomicRmw {
            sz,
            op,
            rs,
            rt,
            rn,
            acquire,
            release,
        } => {
            let base = match op {
                RmwOp::Add => "ldadd",
                RmwOp::Clr => "ldclr",
                RmwOp::Eor => "ldeor",
                RmwOp::Set => "ldset",
                RmwOp::Smax => "ldsmax",
                RmwOp::Smin => "ldsmin",
                RmwOp::Umax => "ldumax",
                RmwOp::Umin => "ldumin",
                RmwOp::Swp => "swp",
            };
            let suffix = match (acquire, release) {
                (false, false) => "",
                (true, false) => "a",
                (false, true) => "l",
                (true, true) => "al",
            };
            let w = match sz {
                MemSz::B => "b",
                MemSz::H => "h",
                _ => "",
            };
            let rsz = if sz == MemSz::D { Sz::X } else { Sz::W };
            format!(
                "{base}{suffix}{w} {}, {}, [{}]",
                reg(rs, rsz),
                reg(rt, rsz),
                reg(rn, Sz::X)
            )
        }
    }
}

/// Spell a register the way its arrangement demands: `v3.4s` for a vector
/// arrangement, and the plain `s3` for a scalar one.
fn areg(v: VReg, arr: VArr) -> String {
    if arr.scalar {
        format!("{}{}", arr.letter(), v.0)
    } else {
        format!("v{}.{}", v.0, arr.suffix())
    }
}

/// The `, #n` a fixed-point conversion carries, and nothing for the plain
/// form.
fn fbits_suffix(fbits: u32) -> String {
    if fbits == 0 {
        String::new()
    } else {
        format!(", #{fbits}")
    }
}

fn lane_letter(lane_bytes: u32) -> char {
    match lane_bytes {
        1 => 'b',
        2 => 'h',
        4 => 's',
        _ => 'd',
    }
}

/// A floating-point immediate the way binutils prints one: C's `%.18e`,
/// with a signed at-least-two-digit exponent.
fn fp_imm_text(v: f64) -> String {
    let s = format!("{v:.18e}");
    let (mantissa, exp) = s.split_once('e').unwrap();
    let (sign, digits) = match exp.strip_prefix('-') {
        Some(d) => ('-', d),
        None => ('+', exp),
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

fn vop3_name(op: VOp3) -> &'static str {
    match op {
        VOp3::Add => "add",
        VOp3::Sub => "sub",
        VOp3::Mul => "mul",
        VOp3::Mla => "mla",
        VOp3::Mls => "mls",
        VOp3::SMax => "smax",
        VOp3::SMin => "smin",
        VOp3::UMax => "umax",
        VOp3::UMin => "umin",
        VOp3::SqAdd => "sqadd",
        VOp3::UqAdd => "uqadd",
        VOp3::SqSub => "sqsub",
        VOp3::UqSub => "uqsub",
        VOp3::SqDmulh => "sqdmulh",
        VOp3::SqRdmulh => "sqrdmulh",
        VOp3::SHAdd => "shadd",
        VOp3::UHAdd => "uhadd",
        VOp3::SrHAdd => "srhadd",
        VOp3::UrHAdd => "urhadd",
        VOp3::SHSub => "shsub",
        VOp3::UHSub => "uhsub",
        VOp3::CmEq => "cmeq",
        VOp3::CmGt => "cmgt",
        VOp3::CmGe => "cmge",
        VOp3::CmHi => "cmhi",
        VOp3::CmHs => "cmhs",
        VOp3::CmTst => "cmtst",
        VOp3::AddP => "addp",
        VOp3::SMaxP => "smaxp",
        VOp3::UMaxP => "umaxp",
        VOp3::SMinP => "sminp",
        VOp3::UMinP => "uminp",
        VOp3::SShl => "sshl",
        VOp3::UShl => "ushl",
        VOp3::FAbd => "fabd",
        VOp3::SAbd => "sabd",
        VOp3::UAbd => "uabd",
        VOp3::FAdd => "fadd",
        VOp3::FSub => "fsub",
        VOp3::FMul => "fmul",
        VOp3::FMulx => "fmulx",
        VOp3::FDiv => "fdiv",
        VOp3::FRecps => "frecps",
        VOp3::FRSqrts => "frsqrts",
        VOp3::FMax => "fmax",
        VOp3::FMin => "fmin",
        VOp3::FMaxNm => "fmaxnm",
        VOp3::FMinNm => "fminnm",
        VOp3::FAddP => "faddp",
        VOp3::FMaxP => "fmaxp",
        VOp3::FMinP => "fminp",
        VOp3::FAcGe => "facge",
        VOp3::FAcGt => "facgt",
        VOp3::FMla => "fmla",
        VOp3::FMls => "fmls",
        VOp3::FCmEq => "fcmeq",
        VOp3::FCmGe => "fcmge",
        VOp3::FCmGt => "fcmgt",
    }
}

fn vop2_name(op: VOp2) -> &'static str {
    match op {
        VOp2::Neg => "neg",
        VOp2::Abs => "abs",
        // binutils prefers the `mvn` alias for the vector bitwise NOT.
        VOp2::Not => "mvn",
        VOp2::Cnt => "cnt",
        VOp2::Saddlp => "saddlp",
        VOp2::Uaddlp => "uaddlp",
        VOp2::Sadalp => "sadalp",
        VOp2::Uadalp => "uadalp",
        VOp2::Clz => "clz",
        VOp2::Cls => "cls",
        VOp2::Rbit => "rbit",
        VOp2::Rev16 => "rev16",
        VOp2::Rev32 => "rev32",
        VOp2::Rev64 => "rev64",
        VOp2::FNeg => "fneg",
        VOp2::FAbs => "fabs",
        VOp2::FSqrt => "fsqrt",
        VOp2::FRecpe => "frecpe",
        VOp2::FRSqrtE => "frsqrte",
        VOp2::FrintN => "frintn",
        VOp2::FrintP => "frintp",
        VOp2::FrintM => "frintm",
        VOp2::FrintZ => "frintz",
        VOp2::FrintA => "frinta",
        VOp2::FrintX => "frintx",
        VOp2::FrintI => "frinti",
        VOp2::CmEqZero => "cmeq",
        VOp2::CmGtZero => "cmgt",
        VOp2::CmGeZero => "cmge",
        VOp2::CmLtZero => "cmlt",
        VOp2::CmLeZero => "cmle",
        VOp2::FCmEqZero => "fcmeq",
        VOp2::FCmGtZero => "fcmgt",
        VOp2::FCmGeZero => "fcmge",
        VOp2::FCmLtZero => "fcmlt",
        VOp2::FCmLeZero => "fcmle",
        VOp2::FcvtZs => "fcvtzs",
        VOp2::FcvtZu => "fcvtzu",
        VOp2::FcvtNs => "fcvtns",
        VOp2::FcvtNu => "fcvtnu",
        VOp2::FcvtPs => "fcvtps",
        VOp2::FcvtPu => "fcvtpu",
        VOp2::FcvtMs => "fcvtms",
        VOp2::FcvtMu => "fcvtmu",
        VOp2::FcvtAs => "fcvtas",
        VOp2::FcvtAu => "fcvtau",
        VOp2::Scvtf => "scvtf",
        VOp2::Ucvtf => "ucvtf",
        VOp2::Xtn => "xtn",
        VOp2::SqXtn => "sqxtn",
        VOp2::UqXtn => "uqxtn",
        VOp2::SqXtun => "sqxtun",
        VOp2::Shll => "shll",
    }
}

/// A SIMD&FP register named by its floating-point view: `s` or `d`.
fn freg(v: VReg, sz: FpSz) -> String {
    format!("{}{}", if sz.is_double() { 'd' } else { 's' }, v.0)
}

/// A SIMD&FP register named by its scalar view: `b`, `h`, `s`, `d` or `q`.
fn vreg_scalar(v: VReg, bytes: u32) -> String {
    let p = match bytes {
        1 => 'b',
        2 => 'h',
        4 => 's',
        8 => 'd',
        _ => 'q',
    };
    format!("{p}{}", v.0)
}

/// A SIMD&FP register named by its vector view, e.g. `v3.16b`.
fn vreg_vector(v: VReg, lane_bytes: u32, q: bool) -> String {
    let lanes = (if q { 16 } else { 8 }) / lane_bytes;
    let t = match lane_bytes {
        1 => 'b',
        2 => 'h',
        4 => 's',
        _ => 'd',
    };
    format!("v{}.{lanes}{t}", v.0)
}

fn load_mnemonic(
    sz: MemSz,
    dest: Sz,
    signed: bool,
    order: Ordering,
    addr: &AddrMode,
) -> &'static str {
    if order == Ordering::Acquire {
        return match sz {
            MemSz::B => "ldarb",
            MemSz::H => "ldarh",
            _ => "ldar",
        };
    }
    if matches!(addr, AddrMode::Literal { .. }) {
        return if signed { "ldrsw" } else { "ldr" };
    }
    match (sz, signed, dest) {
        (MemSz::B, false, _) => "ldrb",
        (MemSz::H, false, _) => "ldrh",
        (MemSz::B, true, _) => "ldrsb",
        (MemSz::H, true, _) => "ldrsh",
        (MemSz::W, true, _) => "ldrsw",
        _ => "ldr",
    }
}

fn mnem_addsub(op: AddOp, set_flags: bool) -> &'static str {
    match (op, set_flags) {
        (AddOp::Add, false) => "add",
        (AddOp::Add, true) => "adds",
        (AddOp::Sub, false) => "sub",
        (AddOp::Sub, true) => "subs",
    }
}

fn fmt_bitfield(sz: Sz, op: BfOp, rd: RegSel, rn: RegSel, immr: u32, imms: u32) -> String {
    let bits = sz.bits();
    let d = reg(rd, sz);
    let n = reg(rn, sz);
    match op {
        BfOp::Ubfm => {
            if imms + 1 == immr {
                return format!("lsl {d}, {n}, #{}", bits - immr);
            }
            if imms == bits - 1 {
                return format!("lsr {d}, {n}, #{immr}");
            }
            if immr == 0 {
                match imms {
                    7 => return format!("uxtb {d}, {}", reg(rn, Sz::W)),
                    15 => return format!("uxth {d}, {}", reg(rn, Sz::W)),
                    _ => {}
                }
            }
            if imms >= immr {
                return format!("ubfx {d}, {n}, #{immr}, #{}", imms - immr + 1);
            }
            format!("ubfiz {d}, {n}, #{}, #{}", bits - immr, imms + 1)
        }
        BfOp::Sbfm => {
            if imms == bits - 1 {
                return format!("asr {d}, {n}, #{immr}");
            }
            if immr == 0 {
                match imms {
                    7 => return format!("sxtb {d}, {}", reg(rn, Sz::W)),
                    15 => return format!("sxth {d}, {}", reg(rn, Sz::W)),
                    31 if sz.is64() => return format!("sxtw {d}, {}", reg(rn, Sz::W)),
                    _ => {}
                }
            }
            if imms >= immr {
                return format!("sbfx {d}, {n}, #{immr}, #{}", imms - immr + 1);
            }
            format!("sbfiz {d}, {n}, #{}, #{}", bits - immr, imms + 1)
        }
        BfOp::Bfm => {
            if imms >= immr {
                if rn == RegSel::Zr {
                    return format!("bfc {d}, #{immr}, #{}", imms - immr + 1);
                }
                return format!("bfxil {d}, {n}, #{immr}, #{}", imms - immr + 1);
            }
            format!("bfi {d}, {n}, #{}, #{}", bits - immr, imms + 1)
        }
    }
}

fn sysreg_name(s: SysReg) -> String {
    match s {
        SysReg::TpidrEl0 => "tpidr_el0".into(),
        SysReg::TpidrroEl0 => "tpidrro_el0".into(),
        SysReg::Cntvct => "cntvct_el0".into(),
        SysReg::Cntfrq => "cntfrq_el0".into(),
        SysReg::Ctr => "ctr_el0".into(),
        SysReg::Dczid => "dczid_el0".into(),
        SysReg::Midr => "midr_el1".into(),
        SysReg::Fpcr => "fpcr".into(),
        SysReg::Fpsr => "fpsr".into(),
        SysReg::Nzcv => "nzcv".into(),
        SysReg::Other(v) => {
            let op0 = (v >> 14) & 3;
            let op1 = (v >> 11) & 7;
            let crn = (v >> 7) & 15;
            let crm = (v >> 3) & 15;
            let op2 = v & 7;
            format!("s{op0}_{op1}_c{crn}_c{crm}_{op2}")
        }
    }
}
