//! Batched Markov top-k reference for the PCTree planner (Phase 2).
//!
//! The incremental planner (`pctree.plan`) scores one frontier parent at a
//! time. This module provides the dense, batched form of the SAME math —
//! `Z[parent, j] = L[j] + Markov(parent)` materialized as a [|parents|, vocab]
//! block per base position — and per-row top-k selection. It is the CPU
//! reference that the DSV4_MINI oracle's MLX-side batched kernel (Phase 3)
//! must agree with byte-for-byte.
//!
//! Agreement contract pins:
//!   - frontier=1, k=1: planner k=1 chain == batched block's per-row argmax
//!     (byte-equal caller-visible token ids; the z = L + M f32 adds are
//!     elementwise and deterministic on both sides).
//!   - k>1: every planner-retained child is a member of its parent's LOCAL
//!     top-k in the batched block (the planner's local cut must match the
//!     dense per-row cut).
//!
//! `k=1` uses the runtime's raw first-maximal strict-greater scan, including
//! its row-zero-NaN behavior.  `k>1` rejects non-finite logits because DSpark
//! has no source-faithful top-k ordering for them.
//!
//! std-only: `zig test src/pctree_batch.zig`.

const std = @import("std");
const pctree = @import("pctree.zig");

pub const BatchError = error{
    /// out.len must equal row_count * vocab.
    InvalidBlock,
    /// rows/row_count/top_k/k must be consistent and >= 1.
    InvalidArgs,
    /// A branch row is non-finite and therefore has no source-faithful top-k.
    NonFiniteBranchLogit,
};

/// Materialize Z[parent, j] = base[j] + Markov(parent) for every parent, one
/// row per parent, row-major [row_count][vocab]. `scorer` must fill the row
/// for each parent token (same contract as the planner).
pub fn buildZBlock(
    allocator: std.mem.Allocator,
    vocab: usize,
    base_row: []const f32,
    scorer: pctree.MarkovScorer,
    parents: []const u32,
    out: []f32,
) anyerror!void {
    if (vocab == 0 or parents.len == 0 or base_row.len != vocab) return error.InvalidBlock;
    const output_len = std.math.mul(usize, parents.len, vocab) catch return error.InvalidBlock;
    if (out.len != output_len) return error.InvalidBlock;
    var buf: [4096]f32 = undefined;
    const row_buf = if (vocab > buf.len) try allocator.alloc(f32, vocab) else buf[0..vocab];
    defer if (vocab > buf.len) allocator.free(row_buf);
    for (parents, 0..) |parent, pi| {
        try scorer.eval(scorer.ctx, parent, row_buf);
        for (0..vocab) |j| out[pi * vocab + j] = base_row[j] + row_buf[j];
    }
}

/// Per-row top-k with first-maximal tie-break. `out.len` must equal
/// row_count * k; trailing slots of a row are repeated on ties shorter than
/// k (a row has exactly `k` output slots, the first `min(k, vocab)` are
/// populated by ranked picks; the rest equal the last pick).
pub fn topKPerRow(
    allocator: std.mem.Allocator,
    zblock: []const f32,
    row_count: usize,
    vocab: usize,
    k: usize,
    out: []u32,
) (BatchError || std.mem.Allocator.Error)!void {
    if (row_count < 1 or vocab < 1 or k < 1) return error.InvalidArgs;
    const block_len = std.math.mul(usize, row_count, vocab) catch return error.InvalidArgs;
    const output_len = std.math.mul(usize, row_count, k) catch return error.InvalidArgs;
    if (zblock.len != block_len) return error.InvalidBlock;
    if (out.len != output_len) return error.InvalidArgs;

    if (k == 1) {
        for (0..row_count) |row| {
            const row_z = zblock[row * vocab ..][0..vocab];
            out[row] = pctree.serialArgmax(row_z);
        }
        return;
    }

    const order = try allocator.alloc(u32, vocab);
    defer allocator.free(order);

    for (0..row_count) |row| {
        const row_z = zblock[row * vocab ..][0..vocab];
        for (row_z) |value| {
            if (!std.math.isFinite(value)) return error.NonFiniteBranchLogit;
        }
        for (order, 0..) |*o, j| o.* = @intCast(j);
        std.mem.sort(u32, order, row_z, struct {
            fn lt(ctx: []const f32, a: u32, b: u32) bool {
                if (ctx[a] != ctx[b]) return ctx[a] > ctx[b];
                return a < b;
            }
        }.lt);
        const picks = @min(k, vocab);
        for (0..picks) |p| out[row * k + p] = order[p];
        for (picks..k) |p| out[row * k + p] = order[picks - 1];
    }
}

/// Top-1 per row (first-max) — the exact serial greedy per parent.
pub fn top1PerRow(
    allocator: std.mem.Allocator,
    zblock: []const f32,
    row_count: usize,
    vocab: usize,
    out: []u32,
) (BatchError || std.mem.Allocator.Error)!void {
    if (row_count < 1 or vocab < 1) return error.InvalidArgs;
    const block_len = std.math.mul(usize, row_count, vocab) catch return error.InvalidArgs;
    if (zblock.len != block_len) return error.InvalidBlock;
    if (out.len != row_count) return error.InvalidArgs;
    _ = allocator;
    for (0..row_count) |ri| {
        const row = zblock[ri * vocab ..][0..vocab];
        out[ri] = pctree.serialArgmax(row);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const ArrayScorerCtx = pctree.ArrayScorerCtx;

test "batch: frontier=1 k=1 batched top-k equals the planner chain byte-for-byte" {
    const allocator = testing.allocator;
    var rng = std.Random.DefaultPrng.init(0xBA7C_A1F2);

    for (0..50) |_| {
        const b = 1 + rng.random().intRangeAtMost(usize, 0, 10);
        const vocab = 4 + rng.random().intRangeAtMost(usize, 0, 7);
        const base = try allocator.alloc(f32, b * vocab);
        defer allocator.free(base);
        const markov = try allocator.alloc(f32, vocab * vocab);
        defer allocator.free(markov);
        for (base) |*v| v.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -8, 8)));
        for (markov) |*v| v.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -8, 8)));
        const root = rng.random().intRangeAtMost(u32, 0, @intCast(vocab - 1));

        var holder: ArrayScorerCtx = undefined;
        const scorer = pctree.newArrayScorer(markov, vocab, &holder);

        // The incremental planner's k=1 chain.
        var p = try pctree.plan(allocator, b, 1, b + 1, vocab, base, scorer, root);
        defer p.deinit();

        // The batched form: per round, one parent (the previous chain token)
        // and one selected child.
        const zblock = try allocator.alloc(f32, vocab);
        defer allocator.free(zblock);
        const chain = p.flat_tokens; // [root, t1, ..., tb]
        const picks = try allocator.alloc(u32, b);
        defer allocator.free(picks);
        for (0..b) |d| {
            const parents = [_]u32{chain[d]};
            try buildZBlock(allocator, vocab, base[d * vocab ..][0..vocab], scorer, &parents, zblock);
            try top1PerRow(allocator, zblock, 1, vocab, picks[d .. d + 1]);
        }
        // batched != planner is a byte-level failure
        try testing.expectEqualSlices(u32, chain[1..], picks);
    }
}

test "batch: every planner-retained child is in its parent's batched local top-k" {
    const allocator = testing.allocator;
    var rng = std.Random.DefaultPrng.init(0x70DD_5EED);

    for (0..40) |_| {
        const b = 1 + rng.random().intRangeAtMost(usize, 0, 6);
        const vocab = 4 + rng.random().intRangeAtMost(usize, 0, 7);
        const k = 1 + rng.random().intRangeAtMost(usize, 0, 3);
        const n = 1 + rng.random().intRangeAtMost(usize, 0, 20);
        const base = try allocator.alloc(f32, b * vocab);
        defer allocator.free(base);
        const markov = try allocator.alloc(f32, vocab * vocab);
        defer allocator.free(markov);
        for (base) |*v| v.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -8, 8)));
        for (markov) |*v| v.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -8, 8)));
        const root = rng.random().intRangeAtMost(u32, 0, @intCast(vocab - 1));

        var holder: ArrayScorerCtx = undefined;
        const scorer = pctree.newArrayScorer(markov, vocab, &holder);
        var p = try pctree.plan(allocator, b, k, n, vocab, base, scorer, root);
        defer p.deinit();

        const zblock = try allocator.alloc(f32, vocab);
        defer allocator.free(zblock);
        const topk = try allocator.alloc(u32, k);
        defer allocator.free(topk);

        for (0..p.n_retained) |r| {
            const node = p.nodes[p.retained[r]];
            if (node.depth == 0) continue;
            // LOCAL top-k of the parent's row at this node's round.
            const parents = [_]u32{p.nodes[node.parent].token};
            try buildZBlock(allocator, vocab, base[@as(usize, node.round) * vocab ..][0..vocab], scorer, &parents, zblock);
            try topKPerRow(allocator, zblock, 1, vocab, k, topk);
            var found = false;
            for (topk) |t| {
                if (t == node.token) {
                    found = true;
                    break;
                }
            }
            try testing.expect(found);
        }
    }
}

test "batch: raw top-1 preserves one-ULP and row-zero-NaN DSpark semantics" {
    const allocator = testing.allocator;
    const vocab: usize = 4;
    const one = @as(f32, @bitCast(@as(u32, 0x3f80_0000)));
    const one_ulp_up = @as(f32, @bitCast(@as(u32, 0x3f80_0001)));
    var zblock = [_]f32{ one, one_ulp_up, -2, -3 };
    var out: [1]u32 = undefined;
    try top1PerRow(allocator, &zblock, 1, vocab, &out);
    try testing.expectEqual(@as(u32, 1), out[0]);
    zblock = .{ std.math.nan(f32), 10, 9, 8 };
    try top1PerRow(allocator, &zblock, 1, vocab, &out);
    try testing.expectEqual(@as(u32, 0), out[0]);
    var topk: [2]u32 = undefined;
    try testing.expectError(error.NonFiniteBranchLogit, topKPerRow(allocator, &zblock, 1, vocab, 2, &topk));
}

test "batch: zero and overflow-shaped dimensions fail closed" {
    const allocator = testing.allocator;
    var out: [1]u32 = undefined;
    try testing.expectError(error.InvalidArgs, top1PerRow(allocator, &.{}, 0, 1, &out));
    try testing.expectError(error.InvalidArgs, topKPerRow(allocator, &.{}, 1, 0, 1, &out));
    try testing.expectError(error.InvalidBlock, buildZBlock(allocator, 0, &.{}, undefined, &.{}, &.{}));
}
