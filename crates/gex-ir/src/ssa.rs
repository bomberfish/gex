//! SSA across blocks for guest registers and flags ([`Form::Ssa`]).
//!
//! [`to_ssa`] turns a memory-form [`Function`] into SSA form: a register
//! read whose value is already known (because the function wrote it, read it
//! earlier, or it flows in from every predecessor) is replaced by that
//! value, and registers whose value differs by path become block
//! parameters. Every `WriteReg` stays (see the [`crate::func`] module
//! documentation), so the context is coherent everywhere and a `ReadReg`
//! that remains is always valid.
//!
//! Where the value of a register is not known, it is read from the context:
//! at function entry, after a clobbering host call, after a syscall, and
//! after a call for every register the callee is not proved to preserve
//! ([`CallEffect`]). A block entered from outside the function (the entry,
//! or one of [`Function::entries`]) starts with each parameter set to the
//! context's value of its register.
//!
//! A [`Stmt::Trap`] needs nothing here: it reads every register when
//! taken, and since every `WriteReg` is kept the context is exact at it.
//!
//! Construction is Cytron et al.'s: parameters at the iterated dominance
//! frontier of each register's definitions, pruned by liveness, then renamed
//! along the dominator tree, then trivial parameters removed.

use crate::func::*;
use crate::*;
use std::collections::HashMap;

/// What a call does to the registers, as SSA construction needs to know it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CallEffect {
    /// Registers the callee is proved to hand back unchanged. Their values
    /// flow through the call; every other register is re-read afterwards.
    pub preserved: RegSet,
    /// The stack pointer after the call minus before it, when known (the
    /// callee's [`Summary::sp_delta`]). The stack pointer is then computed
    /// rather than re-read, which is what lets stack slots be tracked across
    /// calls.
    pub sp_delta: Option<i64>,
}

impl CallEffect {
    /// A call nothing is known about: every register may change.
    pub const UNKNOWN: CallEffect = CallEffect {
        preserved: RegSet::EMPTY,
        sp_delta: None,
    };

    pub fn from_summary(s: &Summary) -> CallEffect {
        CallEffect {
            preserved: s.preserved,
            sp_delta: s.sp_delta,
        }
    }
}

fn clobbering(s: &Stmt) -> bool {
    match s {
        Stmt::HostCall { func, .. }
        | Stmt::Def {
            expr: Expr::HostCall { func, .. },
            ..
        } => func.clobbers_guest_state(),
        _ => false,
    }
}

/// Split each call and syscall continuation that is a join (more than one
/// predecessor, or an entry block) with a synthetic landing block, so the
/// registers a call clobbers can be re-read on the landing's side.
fn split_continuations(f: &mut Function) {
    let preds = f.preds();
    let entries = f.entry_blocks();
    let mut new_blocks: Vec<FBlock> = Vec::new();
    let base = f.blocks.len() as u32;
    for b in f.blocks.iter_mut() {
        let (pc, e) = match &mut b.term {
            FTerm::Call {
                ret: Some(e),
                ret_pc,
                ..
            } => (*ret_pc, e),
            FTerm::Syscall {
                next: Some(e),
                next_pc,
                ..
            } => (*next_pc, e),
            _ => continue,
        };
        if preds[e.to.index()].len() <= 1 && !entries.contains(&e.to) {
            continue;
        }
        let landing = BlockId(base + new_blocks.len() as u32);
        new_blocks.push(FBlock::synthetic(pc, FTerm::Jump(Edge::to(e.to))));
        e.to = landing;
    }
    f.blocks.extend(new_blocks);
}

/// Per-block register effects: `uses` are read before any write or clobber
/// in the block; `kills` are written or clobbered anywhere in it.
fn local_sets(b: &FBlock) -> (RegSet, RegSet) {
    let (mut uses, mut kills) = (RegSet::EMPTY, RegSet::EMPTY);
    for s in &b.stmts {
        if clobbering(s) {
            kills = RegSet::ALL;
            continue;
        }
        match s {
            Stmt::Def {
                expr: Expr::ReadReg(r),
                ..
            } => {
                if !kills.contains(*r) {
                    uses.insert(*r);
                }
            }
            Stmt::WriteReg { reg, .. } => kills.insert(*reg),
            _ => {}
        }
    }
    (uses, kills)
}

/// Convert a memory-form function to SSA form. `call` says what each call
/// preserves; pass `|_| CallEffect::UNKNOWN` when nothing is known.
///
/// The stack pointer is taken to be [`GuestReg::Sp`] (AArch64). An x86-64
/// function keeps `rsp` in `X(4)`: use [`to_ssa_sp`] with that register, or
/// [`CallEffect::sp_delta`] is applied to a register the code never uses
/// and `rsp` is re-read after every call (sound, but less precise).
pub fn to_ssa(f: &Function, call: &dyn Fn(Callee) -> CallEffect) -> Function {
    to_ssa_sp(f, call, GuestReg::Sp)
}

/// [`to_ssa`] with the guest's stack-pointer register given: `sp` is the
/// register a call's [`CallEffect::sp_delta`] applies to.
pub fn to_ssa_sp(f: &Function, call: &dyn Fn(Callee) -> CallEffect, sp: GuestReg) -> Function {
    assert_eq!(f.form, Form::Memory, "to_ssa expects a memory-form function");
    let spi = reg_index(sp);
    let mut f = f.clone();
    split_continuations(&mut f);
    let n = f.blocks.len();
    let entries = f.entry_blocks();
    let is_entry = {
        let mut v = vec![false; n];
        for e in &entries {
            v[e.index()] = true;
        }
        v
    };
    let preds = f.preds();
    let idom = f.idoms();
    let rpo = f.rpo();

    // What each call terminator preserves, looked up once.
    let effects: Vec<Option<CallEffect>> = f
        .blocks
        .iter()
        .map(|b| match b.term {
            FTerm::Call { callee, .. } => Some(call(callee)),
            FTerm::Syscall { .. } => Some(CallEffect::UNKNOWN),
            _ => None,
        })
        .collect();

    // ---- liveness (upward-exposed register reads) -------------------------
    let local: Vec<(RegSet, RegSet)> = f.blocks.iter().map(local_sets).collect();
    let mut live_in = vec![RegSet::EMPTY; n];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().rev() {
            let blk = f.block(b);
            let mut out = RegSet::EMPTY;
            for s in blk.succs() {
                out = out.union(live_in[s.index()]);
            }
            // A call's continuation only needs, from before the call, what
            // flows through it; the rest is re-read after.
            if let Some(eff) = effects[b.index()] {
                out = out.inter(eff.preserved);
                if eff.sp_delta.is_some() {
                    out = out.union(live_in_sp(blk, &live_in, sp));
                }
            }
            let (uses, kills) = local[b.index()];
            let new = uses.union(out.minus(kills));
            if new != live_in[b.index()] {
                live_in[b.index()] = new;
                changed = true;
            }
        }
    }

    // ---- dominance frontiers ---------------------------------------------
    let mut df: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    for &b in &rpo {
        let ps: Vec<BlockId> = preds[b.index()]
            .iter()
            .copied()
            .filter(|p| idom[p.index()].is_some() || is_entry[p.index()])
            .collect();
        // An entry block is also entered from outside: a join with the
        // virtual root even with a single internal predecessor.
        if ps.len() < 2 && !(is_entry[b.index()] && !ps.is_empty()) {
            continue;
        }
        for p in ps {
            let mut runner = Some(p);
            while let Some(r) = runner {
                if Some(r) == idom[b.index()] {
                    break;
                }
                if !df[r.index()].contains(&b) {
                    df[r.index()].push(b);
                }
                runner = idom[r.index()];
            }
        }
    }

    // ---- parameter placement ---------------------------------------------
    // Definition sites of a register: blocks writing or clobbering it, the
    // entry blocks (the context's value), and continuations of calls that
    // do not preserve it.
    let mut def_blocks: Vec<Vec<BlockId>> = vec![Vec::new(); N_REGS];
    for b in f.ids() {
        let (_, kills) = local[b.index()];
        for r in kills.iter() {
            def_blocks[reg_index(r)].push(b);
        }
        if is_entry[b.index()] {
            for i in 0..N_REGS {
                def_blocks[i].push(b);
            }
        }
        if let Some(eff) = effects[b.index()] {
            for s in f.block(b).succs() {
                for i in 0..N_REGS {
                    let r = reg_of_index(i);
                    if !eff.preserved.contains(r) || r == sp {
                        def_blocks[i].push(s);
                    }
                }
            }
        }
    }
    let mut params: Vec<Vec<GuestReg>> = vec![Vec::new(); n];
    for i in 0..N_REGS {
        let r = reg_of_index(i);
        let mut has = vec![false; n];
        let mut work: Vec<BlockId> = def_blocks[i].clone();
        let mut queued = vec![false; n];
        for w in &work {
            queued[w.index()] = true;
        }
        while let Some(x) = work.pop() {
            for &y in &df[x.index()] {
                if has[y.index()] {
                    continue;
                }
                has[y.index()] = true;
                if live_in[y.index()].contains(r) {
                    params[y.index()].push(r);
                }
                if !queued[y.index()] {
                    queued[y.index()] = true;
                    work.push(y);
                }
            }
        }
    }
    for (b, regs) in params.iter().enumerate() {
        let mut ps = Vec::new();
        for &r in regs {
            let v = Value(f.n_values);
            f.n_values += 1;
            ps.push(Param { reg: r, val: v });
        }
        f.blocks[b].params = ps;
    }

    // ---- renaming along the dominator tree -------------------------------
    let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    for &b in &rpo {
        if let Some(p) = idom[b.index()] {
            children[p.index()].push(b);
        }
    }
    // Known value of each register, or None: read the context.
    type State = Vec<Option<Value>>;
    let mut end_state: Vec<Option<State>> = vec![None; n];
    let mut subst: HashMap<Value, Value> = HashMap::new();

    // Iterative preorder over the dominator tree: a tree per entry block,
    // and one per reachable block whose only dominator is the virtual root
    // above the entries (a join of paths from two entries, such as the code
    // after an interior entry that the function's own entry also reaches).
    // Such a block starts with nothing known but its parameters: it is in
    // the dominance frontier of every entry, which defines every register.
    // Left out, its tree kept its memory form, and a jump table ending in it
    // read its index from the context and no longer resolved.
    let mut stack: Vec<BlockId> = entries.iter().rev().copied().collect();
    for &b in rpo.iter().rev() {
        if idom[b.index()].is_none() && !is_entry[b.index()] {
            stack.insert(0, b);
        }
    }
    while let Some(b) = stack.pop() {
        let mut st: State = match idom[b.index()] {
            None => vec![None; N_REGS],
            Some(p) => {
                let mut s = end_state[p.index()].clone().expect("idom visited first");
                // Crossing a call or a syscall: what it does not preserve
                // must be re-read.
                if let Some(eff) = effects[p.index()] {
                    let spv0 = s[spi];
                    for i in 0..N_REGS {
                        if !eff.preserved.contains(reg_of_index(i)) {
                            s[i] = None;
                        }
                    }
                    s[spi] = None;
                    if let (Some(d), Some(spv)) = (eff.sp_delta, spv0) {
                        if d == 0 {
                            s[spi] = Some(spv);
                        } else {
                            let pc = f.block(b).start;
                            let c = f.fresh();
                            let sum = f.fresh();
                            let blk = f.block_mut(b);
                            blk.stmts.insert(
                                0,
                                Stmt::Def {
                                    dst: sum,
                                    expr: Expr::Bin {
                                        op: BinOp::Add,
                                        ty: Ty::I64,
                                        a: spv,
                                        b: c,
                                    },
                                },
                            );
                            blk.stmts.insert(
                                0,
                                Stmt::Def {
                                    dst: c,
                                    expr: Expr::ConstI64(d as u64),
                                },
                            );
                            blk.pcs.insert(0, pc);
                            blk.pcs.insert(0, pc);
                            s[spi] = Some(sum);
                        }
                    }
                }
                s
            }
        };
        for p in &f.block(b).params {
            st[reg_index(p.reg)] = Some(p.val);
        }

        // Statements.
        let blk = f.block_mut(b);
        let old = std::mem::take(&mut blk.stmts);
        let old_pcs = std::mem::take(&mut blk.pcs);
        for (mut s, pc) in old.into_iter().zip(old_pcs) {
            for_each_use_mut(&mut s, |v| {
                if let Some(&w) = subst.get(v) {
                    *v = w;
                }
            });
            if clobbering(&s) {
                blk.stmts.push(s);
                blk.pcs.push(pc);
                st.iter_mut().for_each(|x| *x = None);
                continue;
            }
            match s {
                Stmt::Def {
                    dst,
                    expr: Expr::ReadReg(r),
                } => {
                    if let Some(known) = st[reg_index(r)] {
                        subst.insert(dst, known);
                        continue;
                    }
                    st[reg_index(r)] = Some(dst);
                }
                Stmt::WriteReg { reg, val } => st[reg_index(reg)] = Some(val),
                _ => {}
            }
            blk.stmts.push(s);
            blk.pcs.push(pc);
        }
        blk.term.for_each_use_mut(|v| {
            if let Some(&w) = subst.get(v) {
                *v = w;
            }
        });
        end_state[b.index()] = Some(st);
        for &c in children[b.index()].iter().rev() {
            stack.push(c);
        }
    }

    // ---- edge arguments ---------------------------------------------------
    // Done once every block's end state is known. A register whose value is
    // not known at the end of a predecessor is read there, just before the
    // terminator (never after a call: continuations are not joins).
    for b in f.ids().collect::<Vec<_>>() {
        let Some(mut st) = end_state[b.index()].clone() else {
            continue; // unreachable
        };
        let targets: Vec<BlockId> = f.block(b).succs();
        let mut args: Vec<Vec<Value>> = Vec::new();
        for t in &targets {
            let regs: Vec<GuestReg> = f.block(*t).params.iter().map(|p| p.reg).collect();
            let mut a = Vec::new();
            for r in regs {
                let v = match st[reg_index(r)] {
                    Some(v) => v,
                    None => {
                        let v = f.fresh();
                        let pc = f.block(b).term_pc;
                        f.block_mut(b).push(
                            pc,
                            Stmt::Def {
                                dst: v,
                                expr: Expr::ReadReg(r),
                            },
                        );
                        st[reg_index(r)] = Some(v);
                        v
                    }
                };
                a.push(v);
            }
            args.push(a);
        }
        for (e, a) in f.block_mut(b).term.edges_mut().into_iter().zip(args) {
            e.args = a;
        }
    }

    // Unreachable blocks keep their memory-form bodies, which are valid SSA
    // (block-local values); give their edges arguments by reading.
    for b in 0..n {
        if end_state[b].is_some() {
            continue;
        }
        let targets = f.blocks[b].succs();
        let mut args = Vec::new();
        for t in targets {
            let regs: Vec<GuestReg> = f.block(t).params.iter().map(|p| p.reg).collect();
            let mut a = Vec::new();
            for r in regs {
                let v = f.fresh();
                let pc = f.blocks[b].term_pc;
                f.blocks[b].push(
                    pc,
                    Stmt::Def {
                        dst: v,
                        expr: Expr::ReadReg(r),
                    },
                );
                a.push(v);
            }
            args.push(a);
        }
        for (e, a) in f.blocks[b].term.edges_mut().into_iter().zip(args) {
            e.args = a;
        }
    }

    f.form = Form::Ssa;
    remove_trivial_params(&mut f, &is_entry);
    f
}

/// The stack pointer is live into a call's continuation whenever the
/// continuation reads it, if the call's effect on it is known (it is then
/// computed from the value before the call rather than re-read).
fn live_in_sp(blk: &FBlock, live_in: &[RegSet], sp: GuestReg) -> RegSet {
    let mut out = RegSet::EMPTY;
    for s in blk.succs() {
        if live_in[s.index()].contains(sp) {
            out.insert(sp);
        }
    }
    out
}

/// Remove parameters every predecessor passes the same value for (or the
/// parameter itself, around a loop). Entry blocks keep theirs: the context
/// supplies them on entry from outside.
fn remove_trivial_params(f: &mut Function, is_entry: &[bool]) {
    loop {
        let preds = f.preds();
        let mut replace: Option<(usize, usize, Value, Value)> = None;
        'find: for b in 0..f.blocks.len() {
            if is_entry[b] {
                continue;
            }
            for (pi, p) in f.blocks[b].params.iter().enumerate() {
                let mut same: Option<Value> = None;
                let mut trivial = true;
                // Collect every incoming argument at this position.
                let mut seen_preds = Vec::new();
                for &pr in &preds[b] {
                    if seen_preds.contains(&pr) {
                        continue;
                    }
                    seen_preds.push(pr);
                    for e in f.block(pr).term.edges() {
                        if e.to.index() != b {
                            continue;
                        }
                        let a = e.args[pi];
                        if a == p.val {
                            continue;
                        }
                        match same {
                            None => same = Some(a),
                            Some(s) if s == a => {}
                            Some(_) => {
                                trivial = false;
                            }
                        }
                    }
                }
                if trivial {
                    if let Some(s) = same {
                        replace = Some((b, pi, p.val, s));
                        break 'find;
                    }
                }
            }
        }
        let Some((b, pi, old, new)) = replace else {
            break;
        };
        // Drop the parameter and its argument on every incoming edge.
        f.blocks[b].params.remove(pi);
        for blk in f.blocks.iter_mut() {
            for e in blk.term.edges_mut() {
                if e.to.index() == b {
                    e.args.remove(pi);
                }
            }
        }
        // Replace every use of the old value.
        for blk in f.blocks.iter_mut() {
            for s in blk.stmts.iter_mut() {
                for_each_use_mut(s, |v| {
                    if *v == old {
                        *v = new;
                    }
                });
            }
            blk.term.for_each_use_mut(|v| {
                if *v == old {
                    *v = new;
                }
            });
            for e in blk.term.edges_mut() {
                for a in e.args.iter_mut() {
                    if *a == old {
                        *a = new;
                    }
                }
            }
        }
    }
}
