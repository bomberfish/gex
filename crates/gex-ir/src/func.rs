//! Function-level GIR: a whole guest function as a CFG of lifted blocks.
//!
//! Tier-1 works on one [`Block`](crate::Block) at a time. The function tier
//! (the C backend) works on whole guest functions, and this
//! module is the form it consumes. A [`Function`] is:
//!
//! * a set of [`FBlock`]s indexed by [`BlockId`], block 0 being the primary
//!   entry, each keeping the lifter's statements plus the **guest PC of the
//!   instruction every statement came from** ([`FBlock::pcs`]);
//! * explicit control flow in [`FTerm`]: direct calls with their return
//!   continuation, tail calls, returns, jump tables ([`FTerm::Switch`]) with
//!   their resolved case sets, and indirect jumps nobody could resolve, which
//!   are marked ([`FTerm::Unresolved`]) and never guessed;
//! * optionally **SSA across blocks** ([`Form::Ssa`]): guest registers and
//!   flags flow between blocks as block parameters ([`FBlock::params`]) and
//!   edge arguments ([`Edge::args`]).
//!
//! # The two forms
//!
//! [`Form::Memory`] is what function recovery produces. It has exactly the
//! tier-1 semantics: `ReadReg`/`WriteReg` read and write the architectural
//! state (the context), values are local to the block that defines them, and
//! blocks have no parameters. A backend may keep every guest register in a
//! local and needs nothing more.
//!
//! [`Form::Ssa`] is produced by [`crate::ssa::to_ssa`]. Every `WriteReg` is
//! kept, so **the context is coherent at every program point** exactly as in
//! memory form; what changes is that a register read whose value is already
//! known is replaced by that value, including across blocks through block
//! parameters, and values may be used in any block their definition
//! dominates. A `ReadReg` that remains reads the context where the value is
//! not known (at function entry, after a call that may change the register,
//! after a syscall or a host call that clobbers guest state). A parameter
//! `(r, v)` always equals the context's `r` on entry to its block, which is
//! what a block entered from outside the function starts with. Dropping
//! writes a later reader never observes is an optimization for the backend,
//! using liveness and the call-graph summaries ([`Summary`]); it is not part
//! of the form.
//!
//! # Control flow leaving the function
//!
//! Every transfer out of a function is a terminator of its own: a
//! conditional branch always has two internal edges, and a conditional tail
//! call gets a synthetic block ([`FBlock::synthetic`]) ending in
//! [`FTerm::TailCall`]. A backend therefore writes state back, or checks,
//! only at terminators, never on edges.

use crate::{GuestReg, HostFunc, MemSz, Order, Stmt, Value};
use std::collections::BTreeMap;
use std::fmt;

/// A block's index in [`Function::blocks`]. Block 0 is the primary entry.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.0)
    }
}

impl BlockId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Which of the two forms a function is in; see the module documentation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Form {
    /// Tier-1 semantics; no block parameters; values are block-local.
    Memory,
    /// Registers and flags flow across blocks as block parameters.
    Ssa,
}

/// A recovered guest function.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Function {
    /// The primary entry point: the address callers call.
    pub entry: u64,
    /// The symbol naming `entry`, when there is one.
    pub name: Option<String>,
    /// Other addresses in this function that code outside it is known to
    /// enter at: a second symbol inside the body, a data pointer into it, a
    /// jump from another function into its middle (a shared tail). Each has a
    /// block starting there. The function tier gives each an entry thunk or
    /// leaves it to tier-1; either way it is never silently entered at
    /// `entry`.
    pub entries: Vec<u64>,
    /// The blocks, indexed by [`BlockId`]; `blocks[0]` starts at `entry`.
    pub blocks: Vec<FBlock>,
    /// Values are numbered `0..n_values` across the whole function.
    pub n_values: u32,
    pub form: Form,
}

/// One basic block of a [`Function`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FBlock {
    /// Guest address of the first instruction. For a synthetic block, the
    /// address of the instruction whose edge it sits on.
    pub start: u64,
    /// One past the last instruction; equal to `start` for a synthetic block.
    pub end: u64,
    /// Made by function formation rather than lifted: an exit stub on one
    /// side of a conditional branch, a call's landing block, an entry stub.
    /// It never has a dispatch-table entry of its own.
    pub synthetic: bool,
    /// [`Form::Ssa`] only: the value each listed register holds on entry.
    /// Every edge into the block passes one argument per parameter, in order.
    pub params: Vec<Param>,
    pub stmts: Vec<Stmt>,
    /// `pcs[i]` is the guest address of the instruction `stmts[i]` was
    /// lifted from. Always as long as `stmts`.
    pub pcs: Vec<u64>,
    /// The address of every guest instruction in the block, in order,
    /// including ones that lifted to no statement (a `nop`).
    pub insts: Vec<u64>,
    pub term: FTerm,
    /// The guest address of the instruction that ends the block. For a block
    /// cut short by the instruction budget or by reaching another block,
    /// `end` (the fall-through address).
    pub term_pc: u64,
}

/// A block parameter: register `reg` holds `val` on entry to the block.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Param {
    pub reg: GuestReg,
    pub val: Value,
}

/// A control-flow edge inside the function.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Edge {
    pub to: BlockId,
    /// One per parameter of `to` (empty in memory form).
    pub args: Vec<Value>,
}

impl Edge {
    pub fn to(to: BlockId) -> Edge {
        Edge {
            to,
            args: Vec::new(),
        }
    }
}

/// Where a call or tail call goes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Callee {
    /// A known guest address.
    Direct(u64),
    /// A computed guest address.
    Indirect(Value),
}

/// How a block of a [`Function`] ends.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FTerm {
    /// Continue at another block of this function.
    Jump(Edge),
    /// Two-way branch on an `I32` boolean. Both sides are internal.
    Branch {
        cond: Value,
        taken: Edge,
        not_taken: Edge,
    },
    /// A jump table: `target` is the computed guest address. When it equals
    /// a case's address, control continues at that case's edge. Any other
    /// value leaves the function through the dispatch table (the backend's
    /// default case), which only happens if `table.complete` is false or the
    /// table was modified at run time.
    Switch {
        target: Value,
        /// `(guest address, edge)`, sorted by address, no duplicates.
        cases: Vec<(u64, Edge)>,
        table: JumpTable,
    },
    /// A call (`bl`, `blr`, x86 `call`). The block's statements already did
    /// what the instruction does to guest state (x30, or the pushed return
    /// address), so the callee is entered exactly as tier-1 would enter it.
    Call {
        callee: Callee,
        /// Where the callee is expected to return to.
        ret_pc: u64,
        /// The continuation, `None` when the callee is known not to return.
        /// A backend must still handle a return that does happen (by leaving
        /// to `ret_pc` through the dispatch table), never assume it away.
        ret: Option<Edge>,
    },
    /// Leave the function by a jump that is not a return: a tail call to
    /// another function, a branch into another function's body, or running
    /// off the end into the next function.
    TailCall { callee: Callee },
    /// `ret`: an indirect jump to `target`, expected to be this function's
    /// return address.
    Return { target: Value },
    /// An indirect jump that could not be resolved to a jump table or shown
    /// to be a tail call. Marked, never guessed: it may go anywhere,
    /// including back into this function.
    Unresolved { target: Value },
    /// Enter the kernel. `next` is `None` when the syscall is known not to
    /// return (`exit`, `exit_group`); as with calls, a backend still handles
    /// an unexpected return by leaving to `next_pc`. `imm` is the
    /// system-call number when the ISA encodes it in the instruction (see
    /// [`Term::Syscall`]); 0 when it comes from a register.
    Syscall {
        next_pc: u64,
        next: Option<Edge>,
        imm: u32,
    },
    /// An instruction gex cannot execute: raises SIGILL.
    Illegal { raw: u32, pc: u64 },
}

/// A recovered jump table, attached to [`FTerm::Switch`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct JumpTable {
    /// Address of the table's first entry.
    pub base: u64,
    /// Bytes per entry: 1, 2, 4 or 8.
    pub entry_bytes: u32,
    /// Number of entries the bound check admits.
    pub count: u32,
    /// Where the index comes from, for the printer and for debugging.
    pub index: Option<GuestReg>,
    /// Proven: the bound check dominates the jump, constrains exactly the
    /// bits the target depends on, and the table lies in read-only memory.
    /// When false the cases are a best effort and the default edge is live.
    pub complete: bool,
}

impl FTerm {
    /// Visit every value the terminator reads (not edge arguments).
    pub fn for_each_use(&self, mut f: impl FnMut(Value)) {
        match self {
            FTerm::Branch { cond, .. } => f(*cond),
            FTerm::Switch { target, .. }
            | FTerm::Return { target }
            | FTerm::Unresolved { target } => f(*target),
            FTerm::Call {
                callee: Callee::Indirect(v),
                ..
            }
            | FTerm::TailCall {
                callee: Callee::Indirect(v),
            } => f(*v),
            _ => {}
        }
    }

    /// Mutable access to every value the terminator reads (not edge
    /// arguments).
    pub fn for_each_use_mut(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            FTerm::Branch { cond, .. } => f(cond),
            FTerm::Switch { target, .. }
            | FTerm::Return { target }
            | FTerm::Unresolved { target } => f(target),
            FTerm::Call {
                callee: Callee::Indirect(v),
                ..
            }
            | FTerm::TailCall {
                callee: Callee::Indirect(v),
            } => f(v),
            _ => {}
        }
    }

    /// The edges to other blocks of this function, in a fixed order.
    pub fn edges(&self) -> Vec<&Edge> {
        match self {
            FTerm::Jump(e) => vec![e],
            FTerm::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            FTerm::Switch { cases, .. } => cases.iter().map(|(_, e)| e).collect(),
            FTerm::Call { ret: Some(e), .. } | FTerm::Syscall { next: Some(e), .. } => vec![e],
            _ => Vec::new(),
        }
    }

    pub fn edges_mut(&mut self) -> Vec<&mut Edge> {
        match self {
            FTerm::Jump(e) => vec![e],
            FTerm::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            FTerm::Switch { cases, .. } => cases.iter_mut().map(|(_, e)| e).collect(),
            FTerm::Call { ret: Some(e), .. } | FTerm::Syscall { next: Some(e), .. } => vec![e],
            _ => Vec::new(),
        }
    }

    /// True when control can leave the function here (to a callee, the
    /// caller, the kernel, or anywhere). A backend makes guest-visible state
    /// architecturally correct before every such terminator,
    /// except where call-graph summaries prove it unnecessary.
    pub fn leaves(&self) -> bool {
        !matches!(self, FTerm::Jump(_) | FTerm::Branch { .. })
            && !matches!(self, FTerm::Switch { table, .. } if table.complete)
    }
}

impl FBlock {
    /// An empty block ending in `term`, for synthetic blocks.
    pub fn synthetic(pc: u64, term: FTerm) -> FBlock {
        FBlock {
            start: pc,
            end: pc,
            synthetic: true,
            params: Vec::new(),
            stmts: Vec::new(),
            pcs: Vec::new(),
            insts: Vec::new(),
            term,
            term_pc: pc,
        }
    }

    /// Successor block ids, in the terminator's edge order (may repeat).
    pub fn succs(&self) -> Vec<BlockId> {
        self.term.edges().into_iter().map(|e| e.to).collect()
    }

    /// Append a statement attributed to guest address `pc`.
    pub fn push(&mut self, pc: u64, s: Stmt) {
        self.stmts.push(s);
        self.pcs.push(pc);
    }
}

/// A call or tail call made by a function.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CallSite {
    pub block: BlockId,
    /// The calling instruction.
    pub pc: u64,
    pub callee: Callee,
    /// A tail call ([`FTerm::TailCall`]) rather than a call.
    pub tail: bool,
}

impl Function {
    pub fn block(&self, id: BlockId) -> &FBlock {
        &self.blocks[id.index()]
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut FBlock {
        &mut self.blocks[id.index()]
    }

    pub fn ids(&self) -> impl Iterator<Item = BlockId> {
        (0..self.blocks.len() as u32).map(BlockId)
    }

    /// Allocate a fresh value number.
    pub fn fresh(&mut self) -> Value {
        let v = Value(self.n_values);
        self.n_values += 1;
        v
    }

    /// The lifted (non-synthetic) block starting at `pc`.
    pub fn block_at(&self, pc: u64) -> Option<BlockId> {
        self.blocks
            .iter()
            .position(|b| !b.synthetic && b.start == pc)
            .map(|i| BlockId(i as u32))
    }

    /// Guest start address to block, for the lifted blocks.
    pub fn pc_map(&self) -> BTreeMap<u64, BlockId> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.synthetic)
            .map(|(i, b)| (b.start, BlockId(i as u32)))
            .collect()
    }

    /// The block each entry point starts at: `entry` first, then `entries`.
    pub fn entry_blocks(&self) -> Vec<BlockId> {
        let mut out = vec![BlockId(0)];
        for &e in &self.entries {
            if let Some(b) = self.block_at(e) {
                if !out.contains(&b) {
                    out.push(b);
                }
            }
        }
        out
    }

    /// Predecessors of every block, one entry per edge (a block branching
    /// twice to the same target appears twice).
    pub fn preds(&self) -> Vec<Vec<BlockId>> {
        let mut preds = vec![Vec::new(); self.blocks.len()];
        for id in self.ids() {
            for s in self.block(id).succs() {
                preds[s.index()].push(id);
            }
        }
        preds
    }

    /// Blocks in reverse postorder from the entry blocks. Blocks no entry
    /// reaches are left out.
    pub fn rpo(&self) -> Vec<BlockId> {
        let n = self.blocks.len();
        let mut seen = vec![false; n];
        let mut post = Vec::with_capacity(n);
        for root in self.entry_blocks() {
            if seen[root.index()] {
                continue;
            }
            // Iterative DFS: (block, next successor index).
            let mut stack: Vec<(BlockId, usize)> = vec![(root, 0)];
            seen[root.index()] = true;
            while let Some(&mut (b, ref mut i)) = stack.last_mut() {
                let succs = self.block(b).succs();
                if *i < succs.len() {
                    let s = succs[*i];
                    *i += 1;
                    if !seen[s.index()] {
                        seen[s.index()] = true;
                        stack.push((s, 0));
                    }
                } else {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        post
    }

    /// Immediate dominators, with a virtual root above every entry block.
    /// `idom[b]` is `None` for the entry blocks and for unreachable blocks.
    /// (Cooper, Harvey and Kennedy's iterative algorithm.)
    pub fn idoms(&self) -> Vec<Option<BlockId>> {
        let rpo = self.rpo();
        let n = self.blocks.len();
        const UNDEF: usize = usize::MAX;
        const ROOT: usize = usize::MAX - 1;
        let mut order = vec![UNDEF; n];
        for (i, b) in rpo.iter().enumerate() {
            order[b.index()] = i;
        }
        let preds = self.preds();
        let entries = self.entry_blocks();
        // idom as an rpo position, or ROOT for the virtual root.
        let mut idom = vec![UNDEF; n];
        for e in &entries {
            idom[e.index()] = ROOT;
        }
        let intersect = |idom: &[usize], mut a: usize, mut b: usize| -> usize {
            // a, b are rpo positions or ROOT.
            while a != b {
                if a == ROOT || b == ROOT {
                    return ROOT;
                }
                while a > b {
                    a = idom[rpo[a].index()];
                    if a == ROOT {
                        return ROOT;
                    }
                }
                while b > a {
                    b = idom[rpo[b].index()];
                    if b == ROOT {
                        return ROOT;
                    }
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &rpo {
                if entries.contains(&b) {
                    continue;
                }
                let mut new = UNDEF;
                for p in &preds[b.index()] {
                    let pp = order[p.index()];
                    if pp == UNDEF || idom[p.index()] == UNDEF {
                        continue;
                    }
                    new = if new == UNDEF {
                        pp
                    } else {
                        intersect(&idom, pp, new)
                    };
                }
                if new != UNDEF && idom[b.index()] != new {
                    idom[b.index()] = new;
                    changed = true;
                }
            }
        }
        (0..n)
            .map(|i| match idom[i] {
                UNDEF | ROOT => None,
                p => Some(rpo[p]),
            })
            .collect()
    }

    /// Every call and tail call, in block order.
    pub fn call_sites(&self) -> Vec<CallSite> {
        let mut out = Vec::new();
        for id in self.ids() {
            let b = self.block(id);
            match b.term {
                FTerm::Call { callee, .. } => out.push(CallSite {
                    block: id,
                    pc: b.term_pc,
                    callee,
                    tail: false,
                }),
                FTerm::TailCall { callee } => out.push(CallSite {
                    block: id,
                    pc: b.term_pc,
                    callee,
                    tail: true,
                }),
                _ => {}
            }
        }
        out
    }

    /// Blocks whose terminator leaves the function ([`FTerm::leaves`]).
    pub fn exits(&self) -> Vec<BlockId> {
        self.ids().filter(|&b| self.block(b).term.leaves()).collect()
    }

    /// Guest bytes covered by lifted blocks, merged into disjoint ranges.
    pub fn extent(&self) -> Vec<(u64, u64)> {
        let mut r: Vec<(u64, u64)> = self
            .blocks
            .iter()
            .filter(|b| !b.synthetic && b.end > b.start)
            .map(|b| (b.start, b.end))
            .collect();
        r.sort_unstable();
        let mut out: Vec<(u64, u64)> = Vec::new();
        for (s, e) in r {
            match out.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => out.push((s, e)),
            }
        }
        out
    }

    /// Check the structural invariants, returning the first violation.
    ///
    /// Both forms: `pcs` as long as `stmts`, edges to existing blocks, one
    /// argument per parameter, every value defined once and below
    /// `n_values`. Memory form: no parameters, and every value used in the
    /// block that defines it, after its definition. SSA form: every use
    /// dominated by its definition.
    pub fn verify(&self) -> Result<(), String> {
        let n = self.blocks.len();
        if n == 0 {
            return Err("function has no blocks".into());
        }
        if self.blocks[0].start != self.entry {
            return Err(format!(
                "block 0 starts at {:#x}, entry is {:#x}",
                self.blocks[0].start, self.entry
            ));
        }
        // Where each value is defined: (block, stmt index), stmt index
        // `usize::MAX` for a parameter.
        let mut def: Vec<Option<(usize, usize)>> = vec![None; self.n_values as usize];
        let mut define = |v: Value, at: (usize, usize)| -> Result<(), String> {
            let slot = def
                .get_mut(v.0 as usize)
                .ok_or_else(|| format!("{v:?} is not below n_values {}", self.n_values))?;
            if slot.is_some() {
                return Err(format!("{v:?} is defined twice"));
            }
            *slot = Some(at);
            Ok(())
        };
        for (bi, b) in self.blocks.iter().enumerate() {
            if b.pcs.len() != b.stmts.len() {
                return Err(format!("b{bi}: {} pcs for {} stmts", b.pcs.len(), b.stmts.len()));
            }
            if self.form == Form::Memory && !b.params.is_empty() {
                return Err(format!("b{bi}: parameters in memory form"));
            }
            for p in &b.params {
                define(p.val, (bi, usize::MAX))?;
            }
            for (si, s) in b.stmts.iter().enumerate() {
                let mut err = Ok(());
                for_each_def(s, |v| {
                    if err.is_ok() {
                        err = define(v, (bi, si));
                    }
                });
                err?;
            }
            for e in b.term.edges() {
                let Some(to) = self.blocks.get(e.to.index()) else {
                    return Err(format!("b{bi}: edge to missing {:?}", e.to));
                };
                if e.args.len() != to.params.len() {
                    return Err(format!(
                        "b{bi}: edge to {:?} passes {} args for {} params",
                        e.to,
                        e.args.len(),
                        to.params.len()
                    ));
                }
            }
        }
        let idom = if self.form == Form::Ssa {
            Some(self.idoms())
        } else {
            None
        };
        let dominates = |a: usize, mut b: usize| -> bool {
            let idom = idom.as_ref().unwrap();
            loop {
                if a == b {
                    return true;
                }
                match idom[b] {
                    Some(p) => b = p.index(),
                    None => return false,
                }
            }
        };
        let reachable: Vec<bool> = {
            let mut r = vec![false; n];
            for b in self.rpo() {
                r[b.index()] = true;
            }
            r
        };
        let check = |v: Value, bi: usize, at: usize| -> Result<(), String> {
            let Some(Some((db, ds))) = def.get(v.0 as usize) else {
                return Err(format!("b{bi}: {v:?} is used but never defined"));
            };
            let (db, ds) = (*db, *ds);
            if db == bi {
                // Parameters (ds == MAX) come before every statement.
                if ds != usize::MAX && ds >= at {
                    return Err(format!("b{bi}: {v:?} used before its definition"));
                }
                return Ok(());
            }
            match self.form {
                Form::Memory => Err(format!(
                    "b{bi}: {v:?} is defined in b{db}; memory-form values are block-local"
                )),
                Form::Ssa => {
                    // Uses in unreachable blocks are not checked: nothing
                    // dominates them.
                    if !reachable[bi] || dominates(db, bi) {
                        Ok(())
                    } else {
                        Err(format!("b{bi}: {v:?} (defined in b{db}) does not dominate its use"))
                    }
                }
            }
        };
        for (bi, b) in self.blocks.iter().enumerate() {
            for (si, s) in b.stmts.iter().enumerate() {
                let mut err = Ok(());
                s.for_each_use(|v| {
                    if err.is_ok() {
                        err = check(v, bi, si);
                    }
                });
                err?;
            }
            let end = b.stmts.len();
            let mut err = Ok(());
            b.term.for_each_use(|v| {
                if err.is_ok() {
                    err = check(v, bi, end);
                }
            });
            err?;
            for e in b.term.edges() {
                for &a in &e.args {
                    check(a, bi, end)?;
                }
            }
        }
        Ok(())
    }
}

/// Visit every value a statement defines.
pub fn for_each_def(s: &Stmt, mut f: impl FnMut(Value)) {
    match s {
        Stmt::Def { dst, .. } | Stmt::MemCmp { dst, .. } => f(*dst),
        Stmt::X87 { outs, .. } => outs.iter().copied().for_each(f),
        Stmt::StrOp {
            out_count,
            out_dst,
            out_src,
            out_a,
            out_b,
            ..
        } => {
            f(*out_count);
            f(*out_dst);
            f(*out_src);
            f(*out_a);
            f(*out_b);
        }
        _ => {}
    }
}

/// Mutable access to every value a statement reads.
pub fn for_each_use_mut(s: &mut Stmt, mut f: impl FnMut(&mut Value)) {
    match s {
        Stmt::Def { expr, .. } => expr_uses_mut(expr, &mut f),
        Stmt::WriteReg { val, .. } => f(val),
        Stmt::Store { addr, val, .. } | Stmt::VStore { addr, val, .. } => {
            f(addr);
            f(val);
        }
        Stmt::HostCall { args, .. } => args.iter_mut().for_each(f),
        Stmt::X87 { args, .. } => args.iter_mut().for_each(f),
        Stmt::Trap { cond, .. } => f(cond),
        Stmt::MemCopy { dst, src, len } => {
            f(dst);
            f(src);
            f(len);
        }
        Stmt::MemFill { dst, val, len } => {
            f(dst);
            f(val);
            f(len);
        }
        Stmt::MemCmp { a, b, len, .. } => {
            f(a);
            f(b);
            f(len);
        }
        Stmt::StrOp {
            count,
            dst,
            src,
            val,
            df,
            ..
        } => {
            f(count);
            f(dst);
            f(src);
            f(val);
            f(df);
        }
        Stmt::Fence => {}
    }
}

/// Mutable access to every value a statement defines.
pub fn for_each_def_mut(s: &mut Stmt, mut f: impl FnMut(&mut Value)) {
    match s {
        Stmt::Def { dst, .. } | Stmt::MemCmp { dst, .. } => f(dst),
        Stmt::X87 { outs, .. } => outs.iter_mut().for_each(f),
        Stmt::StrOp {
            out_count,
            out_dst,
            out_src,
            out_a,
            out_b,
            ..
        } => {
            f(out_count);
            f(out_dst);
            f(out_src);
            f(out_a);
            f(out_b);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Memory effects
// ---------------------------------------------------------------------------

/// What a memory access does.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Access {
    Read,
    Write,
    /// An atomic read-modify-write, or a compare-and-swap.
    ReadWrite,
}

/// One guest-memory access made by a statement.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct MemAccess {
    pub access: Access,
    /// The address operand (a full 64-bit guest address).
    pub addr: Value,
    /// Added to `addr`.
    pub offset: i64,
    /// Bytes accessed, `None` when it is a run-time length (`memcpy`,
    /// `rep movs`), in which case the access may extend in either direction
    /// for `StrOp` with an unknown direction flag.
    pub bytes: Option<u32>,
    pub order: Order,
}

/// The side effects of one statement, for alias analysis and code motion.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Effect {
    /// Guest-memory accesses, in execution order.
    pub mem: Vec<MemAccess>,
    /// A memory fence.
    pub fence: bool,
    /// Architectural register read (only `ReadReg`).
    pub reads_reg: Option<GuestReg>,
    /// Architectural register written (only `WriteReg`).
    pub writes_reg: Option<GuestReg>,
    /// A host call; when [`HostFunc::clobbers_guest_state`], it may read and
    /// write every register and all of memory.
    pub host: Option<HostFunc>,
    /// A conditional fault ([`Stmt::Trap`]): when taken, it reads every
    /// register (the context must be exact) and control may unwind; when
    /// not, it does nothing. No memory effect.
    pub trap: bool,
}

impl Effect {
    /// Nothing observable: can be reordered or deleted when unused.
    pub fn is_pure(&self) -> bool {
        self.mem.is_empty()
            && !self.fence
            && self.reads_reg.is_none()
            && self.writes_reg.is_none()
            && self.host.is_none()
            && !self.trap
    }

    /// Reads every register: a host call that may clobber guest state, or a
    /// trap that may be taken.
    pub fn reads_all_state(&self) -> bool {
        self.trap || self.clobbers_all()
    }

    /// May read or write any guest state (a syscall-like host call).
    pub fn clobbers_all(&self) -> bool {
        self.host.is_some_and(|h| h.clobbers_guest_state())
    }
}

fn mem_sz_bytes(sz: MemSz) -> Option<u32> {
    Some(sz.bytes())
}

/// The effects of one statement.
pub fn effect(s: &Stmt) -> Effect {
    use crate::Expr;
    let mut e = Effect::default();
    let rd = |addr: Value, offset: i64, bytes: Option<u32>, order: Order| MemAccess {
        access: Access::Read,
        addr,
        offset,
        bytes,
        order,
    };
    let wr = |addr: Value, offset: i64, bytes: Option<u32>, order: Order| MemAccess {
        access: Access::Write,
        addr,
        offset,
        bytes,
        order,
    };
    match s {
        Stmt::Def { expr, .. } => match expr {
            Expr::ReadReg(r) => e.reads_reg = Some(*r),
            Expr::Load {
                addr,
                offset,
                sz,
                order,
                ..
            } => e.mem.push(rd(*addr, *offset, mem_sz_bytes(*sz), *order)),
            Expr::VLoad {
                addr,
                offset,
                bytes,
            } => e.mem.push(rd(*addr, *offset, Some(*bytes), Order::Relaxed)),
            Expr::Rmw { addr, sz, .. } | Expr::CmpXchg { addr, sz, .. } => {
                e.mem.push(MemAccess {
                    access: Access::ReadWrite,
                    addr: *addr,
                    offset: 0,
                    bytes: mem_sz_bytes(*sz),
                    order: Order::SeqCst,
                })
            }
            Expr::HostCall { func, .. } => e.host = Some(*func),
            _ => {}
        },
        Stmt::WriteReg { reg, .. } => e.writes_reg = Some(*reg),
        Stmt::Store {
            addr,
            offset,
            sz,
            order,
            ..
        } => e.mem.push(wr(*addr, *offset, mem_sz_bytes(*sz), *order)),
        Stmt::VStore {
            addr,
            offset,
            bytes,
            ..
        } => e.mem.push(wr(*addr, *offset, Some(*bytes), Order::Relaxed)),
        Stmt::HostCall { func, .. } => e.host = Some(*func),
        // Pure: no effect beyond the values it defines.
        Stmt::X87 { .. } => {}
        Stmt::Trap { .. } => e.trap = true,
        Stmt::MemCopy { dst, src, .. } => {
            e.mem.push(rd(*src, 0, None, Order::Relaxed));
            e.mem.push(wr(*dst, 0, None, Order::Relaxed));
        }
        Stmt::MemFill { dst, .. } => e.mem.push(wr(*dst, 0, None, Order::Relaxed)),
        Stmt::MemCmp { a, b, .. } => {
            e.mem.push(rd(*a, 0, None, Order::Relaxed));
            e.mem.push(rd(*b, 0, None, Order::Relaxed));
        }
        Stmt::StrOp { kind, dst, src, .. } => {
            use crate::StrKind::*;
            match kind {
                Movs => {
                    e.mem.push(rd(*src, 0, None, Order::Relaxed));
                    e.mem.push(wr(*dst, 0, None, Order::Relaxed));
                }
                Stos => e.mem.push(wr(*dst, 0, None, Order::Relaxed)),
                Scas => e.mem.push(rd(*dst, 0, None, Order::Relaxed)),
                Cmps => {
                    e.mem.push(rd(*src, 0, None, Order::Relaxed));
                    e.mem.push(rd(*dst, 0, None, Order::Relaxed));
                }
            }
        }
        Stmt::Fence => e.fence = true,
    }
    e
}

// ---------------------------------------------------------------------------
// Register sets and call-graph summaries
// ---------------------------------------------------------------------------

/// A set of [`GuestReg`]s, flags included, as a bitset. See [`reg_index`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct RegSet(pub u128);

/// Number of distinct [`GuestReg`] indices.
pub const N_REGS: usize = 83;

/// A dense index for every [`GuestReg`]: `x0`–`x30` are 0–30, `sp` 31,
/// `v0`–`v31` 32–63, then the rest in declaration order.
pub fn reg_index(r: GuestReg) -> usize {
    match r {
        GuestReg::X(n) => n as usize,
        GuestReg::Sp => 31,
        GuestReg::V128(n) => 32 + n as usize,
        GuestReg::Pc => 64,
        GuestReg::N => 65,
        GuestReg::Z => 66,
        GuestReg::C => 67,
        GuestReg::V => 68,
        GuestReg::Tpidr => 69,
        GuestReg::TpidrRo => 70,
        GuestReg::MonAddr => 71,
        GuestReg::MonValue => 72,
        GuestReg::MonSize => 73,
        GuestReg::P => 74,
        GuestReg::A => 75,
        GuestReg::D => 76,
        GuestReg::Mxcsr => 77,
        GuestReg::X87Top => 78,
        GuestReg::X87Cw => 79,
        GuestReg::X87Sw => 80,
        GuestReg::X87Tags => 81,
        GuestReg::CtxAddr => 82,
    }
}

/// The inverse of [`reg_index`].
pub fn reg_of_index(i: usize) -> GuestReg {
    match i {
        0..=30 => GuestReg::X(i as u8),
        31 => GuestReg::Sp,
        32..=63 => GuestReg::V128((i - 32) as u8),
        64 => GuestReg::Pc,
        65 => GuestReg::N,
        66 => GuestReg::Z,
        67 => GuestReg::C,
        68 => GuestReg::V,
        69 => GuestReg::Tpidr,
        70 => GuestReg::TpidrRo,
        71 => GuestReg::MonAddr,
        72 => GuestReg::MonValue,
        73 => GuestReg::MonSize,
        74 => GuestReg::P,
        75 => GuestReg::A,
        76 => GuestReg::D,
        77 => GuestReg::Mxcsr,
        78 => GuestReg::X87Top,
        79 => GuestReg::X87Cw,
        80 => GuestReg::X87Sw,
        81 => GuestReg::X87Tags,
        82 => GuestReg::CtxAddr,
        _ => panic!("register index {i} out of range"),
    }
}

impl RegSet {
    pub const EMPTY: RegSet = RegSet(0);
    /// Every register and flag.
    pub const ALL: RegSet = RegSet((1u128 << N_REGS) - 1);

    pub fn of(regs: &[GuestReg]) -> RegSet {
        let mut s = RegSet::EMPTY;
        for &r in regs {
            s.insert(r);
        }
        s
    }
    pub fn contains(self, r: GuestReg) -> bool {
        self.0 >> reg_index(r) & 1 != 0
    }
    pub fn insert(&mut self, r: GuestReg) {
        self.0 |= 1u128 << reg_index(r);
    }
    pub fn remove(&mut self, r: GuestReg) {
        self.0 &= !(1u128 << reg_index(r));
    }
    pub fn union(self, o: RegSet) -> RegSet {
        RegSet(self.0 | o.0)
    }
    pub fn inter(self, o: RegSet) -> RegSet {
        RegSet(self.0 & o.0)
    }
    pub fn minus(self, o: RegSet) -> RegSet {
        RegSet(self.0 & !o.0)
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }
    pub fn iter(self) -> impl Iterator<Item = GuestReg> {
        (0..N_REGS)
            .filter(move |&i| self.0 >> i & 1 != 0)
            .map(reg_of_index)
    }
}

impl fmt::Debug for RegSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == RegSet::ALL {
            return write!(f, "{{all}}");
        }
        write!(f, "{{")?;
        for (i, r) in self.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", crate::text::reg_name(r))?;
        }
        write!(f, "}}")
    }
}

/// A range of stack offsets relative to the function's entry stack pointer.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Slots {
    None,
    /// `[lo, hi)` bytes relative to the entry stack pointer.
    Range(i64, i64),
    /// Anywhere: an access whose address could not be bounded.
    Unknown,
}

impl Slots {
    pub fn add(self, lo: i64, hi: i64) -> Slots {
        match self {
            Slots::None => Slots::Range(lo, hi),
            Slots::Range(a, b) => Slots::Range(a.min(lo), b.max(hi)),
            Slots::Unknown => Slots::Unknown,
        }
    }
    pub fn join(self, o: Slots) -> Slots {
        match (self, o) {
            (Slots::Unknown, _) | (_, Slots::Unknown) => Slots::Unknown,
            (Slots::None, x) | (x, Slots::None) => x,
            (Slots::Range(a, b), Slots::Range(c, d)) => Slots::Range(a.min(c), b.max(d)),
        }
    }
    /// Shift by `d` bytes (a callee's frame seen from its caller).
    pub fn shift(self, d: i64) -> Slots {
        match self {
            Slots::Range(a, b) => Slots::Range(a + d, b + d),
            x => x,
        }
    }
}

/// What a function (with everything it calls) does, as seen by a caller.
///
/// Every field over-approximates: when anything in the function's call tree
/// is unknown (an unresolved indirect jump, an indirect call, a callee that
/// was not recovered), the summary says "everything" for it. That is
/// [`Summary::unknown`], the value any consumer must assume for a callee it
/// has no summary for.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Summary {
    /// Registers and flags whose entry value may be read (before being
    /// written) by the function or its callees.
    pub inputs: RegSet,
    /// Registers and flags written on every path that returns (by the
    /// function or a callee): after a call, what the caller reads of these
    /// is the callee's result, not the caller's own entry value. A must-set:
    /// empty when anything is unknown.
    pub always_written: RegSet,
    /// Stack bytes, relative to the entry stack pointer, the function may
    /// read from its caller's frame (stack arguments; on x86-64 the return
    /// address at offset 0).
    pub stack_inputs: Slots,
    /// Registers and flags that may hold a different value when the function
    /// returns. The complement of `preserved`.
    pub outputs: RegSet,
    /// Registers and flags proved to hold their entry value at every return:
    /// either never written, or saved and restored with nothing able to
    /// change the saved copy in between. Proved, never assumed from the ABI.
    pub preserved: RegSet,
    /// The stack pointer at every return minus the entry stack pointer, when
    /// it is the same on every return path (AArch64: 0; x86-64: +8, the
    /// return address popped).
    pub sp_delta: Option<i64>,
    /// Stack bytes relative to the entry stack pointer the function may
    /// write *above* its entry stack pointer (into its caller's frame).
    pub caller_frame_writes: Slots,
    /// Registers whose entry values the function (or a callee) may use to
    /// form a memory address. A caller whose frame address sits in one of
    /// these at a call has let its frame escape to the callee. `sp` is in
    /// the set only when the function lets its entry stack pointer (its
    /// caller's frame boundary) escape: stores it, returns something
    /// derived from it, or saves it as `setjmp` does.
    pub addr_inputs: RegSet,
    /// Registers whose entry values the function (or a callee) may put
    /// where other code can read them: memory other than its own frame
    /// while that frame has not escaped, the kernel, or unknown code. A
    /// caller whose frame address sits in one of these at a call has let
    /// its frame escape, as with `addr_inputs`.
    pub escaping_inputs: RegSet,
    /// Registers whose entry values may flow into a register that differs
    /// at return (an output), not counting a register saved and restored in
    /// place. What a caller's value in one of these may reach through the
    /// call's outputs; `sp` here means something derived from the entry
    /// stack pointer may be returned.
    pub output_sources: RegSet,
    /// Some path returns to the caller.
    pub may_return: bool,
    /// May reach an event that needs every function-tier frame unwound:
    /// `fork`, `clone`, `execve`, a signal raised
    /// synchronously, a stack switch (`longjmp`, `setcontext`), or code the
    /// analysis could not see.
    pub may_unwind: bool,
    /// May transfer control somewhere other than its return address: an
    /// unresolved indirect jump, a return through a value not proved to be
    /// the return address, or a callee that may. A caller checks after the
    /// call ("returns are checked").
    pub may_leave_pending: bool,
    /// Some call or jump in the call tree goes to code the analysis does not
    /// know. Everything above is then "everything" for that path.
    pub unknown_callee: bool,
    /// May read the return address its caller left (x86-64: the slot at
    /// the entry stack pointer; AArch64: x30 other than to return through
    /// it or to save it).
    pub reads_return_address: bool,
}

impl Summary {
    /// The summary of code nothing is known about: reads, writes and may do
    /// everything.
    pub fn unknown() -> Summary {
        Summary {
            inputs: RegSet::ALL,
            always_written: RegSet::EMPTY,
            stack_inputs: Slots::Unknown,
            outputs: RegSet::ALL,
            preserved: RegSet::EMPTY,
            sp_delta: None,
            caller_frame_writes: Slots::Unknown,
            addr_inputs: RegSet::ALL,
            escaping_inputs: RegSet::ALL,
            output_sources: RegSet::ALL,
            may_return: true,
            may_unwind: true,
            may_leave_pending: true,
            unknown_callee: true,
            reads_return_address: true,
        }
    }

    /// The optimistic starting point of the fixpoint: nothing read, nothing
    /// written, everything preserved, never returns. Only ever a starting
    /// point; a converged summary is sound.
    pub fn bottom() -> Summary {
        Summary {
            inputs: RegSet::EMPTY,
            always_written: RegSet::ALL,
            stack_inputs: Slots::None,
            outputs: RegSet::EMPTY,
            preserved: RegSet::ALL,
            sp_delta: None,
            caller_frame_writes: Slots::None,
            addr_inputs: RegSet::EMPTY,
            escaping_inputs: RegSet::EMPTY,
            output_sources: RegSet::EMPTY,
            may_return: false,
            may_unwind: false,
            may_leave_pending: false,
            unknown_callee: false,
            reads_return_address: false,
        }
    }
}

/// Mutable access to every value an expression reads.
fn expr_uses_mut(expr: &mut crate::Expr, f: &mut impl FnMut(&mut Value)) {
    use crate::Expr::*;
    match expr {
        ConstI32(_) | ConstI64(_) | ReadReg(_) | VConst(_) => {}
        Bin { a, b, .. } | Cmp { a, b, .. } => {
            f(a);
            f(b);
        }
        Un { a, .. } | Wrap(a) | Extend { a, .. } | ExtendBits { a, .. } => f(a),
        Select { cond, a, b, .. } => {
            f(cond);
            f(a);
            f(b);
        }
        Load { addr, .. } | VLoad { addr, .. } => f(addr),
        Rmw { addr, val, .. } => {
            f(addr);
            f(val);
        }
        CmpXchg {
            addr,
            expected,
            replacement,
            ..
        } => {
            f(addr);
            f(expected);
            f(replacement);
        }
        HostCall { args, .. } => args.iter_mut().for_each(f),
        VSplat { a, .. }
        | VExtract { a, .. }
        | VUn { a, .. }
        | VBitmask { a, .. }
        | VLaneUn { a, .. } => f(a),
        VLaneBin { a, b, .. } | VShuffle { a, b, .. } | VBin { a, b, .. } => {
            f(a);
            f(b);
        }
        VShift { a, amount, .. } => {
            f(a);
            f(amount);
        }
        VSwizzle { a, idx } => {
            f(a);
            f(idx);
        }
        VBitSelect { mask, a, b } => {
            f(mask);
            f(a);
            f(b);
        }
        VCvt { a, b, .. } => {
            f(a);
            if let Some(b) = b {
                f(b);
            }
        }
        VReplace { a, val, .. } => {
            f(a);
            f(val);
        }
        FpBin { a, b, .. } | FpCmp { a, b, .. } => {
            f(a);
            f(b);
        }
        FpUn { a, .. } | FpCvt { a, .. } => f(a),
        Mx { mx, e } => {
            f(mx);
            expr_uses_mut(e, f);
        }
    }
}
