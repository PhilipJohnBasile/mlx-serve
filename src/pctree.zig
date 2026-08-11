//! Pure CPU PCTree planner for DeepSeek V4 DSpark (target-independent).
//!
//! Converts a single block of DSpark base logits plus the pretrained Markov
//! bigram head into a budgeted prefix-closed tree of candidate drafts.
//!
//! This module is std-only and deliberately contains no MLX references so it
//! can be unit-tested in isolation (`zig test src/pctree.zig`). It implements
//! the Phase-1 contract from the PCTree execution brief:
//!
//!   Z[d, parent] = L[d] + Markov(parent)
//!   row-wise log-softmax -> per-parent local top-k
//!   child path score = parent.path_score + child.logprob
//!   retain every local top-k child in the candidate pool
//!   next frontier = best k nodes from the current candidate layer
//!   after all depths: sort by (path desc, depth asc, stable id asc)
//!   retain top N nodes including the root
//!
//! Determinism:
//!   - no randomness anywhere; all reductions run in a fixed ascending-index
//!     order
//!   - k=1 selects directly from the raw f32 `z = L + Markov` row with the
//!     runtime's strict-greater scan.  This is deliberately *before*
//!     log-softmax: normalizing one-ULP-near logits can erase their ordering.
//!     It reproduces DSpark's first-max result even when row zero is NaN.
//!   - branching (`k > 1`) rejects non-finite raw logits.  DSpark defines only
//!     a top-1 scan for such rows; inventing a top-k order would not be a
//!     source-faithful tree policy.
//!
//! Retention is automatically prefix-closed: path scores are non-increasing
//! along the tree (children add logp <= 0), so the top-N by
//! (path desc, depth asc, id asc) always contains every ancestor of every
//! retained node. A defensive closure pass re-asserts this by construction.
//!
//! Output shapes (see `Plan`):
//!   - `retained` / `flat_tokens` in final display order
//!   - `ancestor_mask[child_row][ancestor_row]` includes self (the
//!     self-attention position) and every ancestor
//!   - `retrieve_rows`: one root-to-leaf rank row per retained leaf, in
//!     display order; the winner path is the first row

const std = @import("std");

pub const PlannerError = error{
    /// base_logits.len must be exactly b * vocab.
    InvalidBaseLogits,
    /// b, k, n, vocab must all be >= 1.
    InvalidBudget,
    /// root_token must be a valid vocab index.
    InvalidRootToken,
    /// The structural node bound, input shape, or stable node ids overflow.
    NodeBudgetExceeded,
    /// k>1 has no source-faithful ranking when a raw Markov-conditioned row
    /// contains NaN or infinity.  k==1 still follows DSpark exactly.
    NonFiniteBranchLogit,
    /// The array-backed test scorer does not contain the requested full row.
    InvalidMarkovScorer,
};

/// Host-side Markov bigram scorer. For each token it fills `out[0..vocab]`
/// with the bigram bias row `Markov(token)` exactly as the runtime computes
/// it (matvec on the runtime's own tensors). The planner treats it as an
/// opaque deterministic function so the same binary can run pure-CPU tests
/// (arrays) and the DSV4_MINI oracle (runtime matvec) without a code fork.
pub const MarkovScorer = struct {
    ctx: ?*anyopaque,
    eval: *const fn (ctx: ?*anyopaque, token: u32, out: []f32) anyerror!void,
};

pub const ArrayScorerCtx = struct {
    markov: []const f32,
    vocab: usize,
};

fn arrayEval(ctx: ?*anyopaque, token: u32, out: []f32) anyerror!void {
    const c: *const ArrayScorerCtx = @ptrCast(@alignCast(ctx.?));
    if (out.len != c.vocab) return error.InvalidMarkovScorer;
    const start = std.math.mul(usize, @as(usize, token), c.vocab) catch return error.InvalidMarkovScorer;
    const end = std.math.add(usize, start, c.vocab) catch return error.InvalidMarkovScorer;
    if (end > c.markov.len) return error.InvalidMarkovScorer;
    const row = c.markov[start..end];
    @memcpy(out, row);
}

/// Wrap a row-major [vocab][vocab] f32 matrix as a scorer
/// (markov[parent][child]). Public so model-side oracles and external
/// debuggers can drive the planner without a code fork.
pub fn newArrayScorer(markov: []const f32, vocab: usize, holder: *ArrayScorerCtx) MarkovScorer {
    holder.* = .{ .markov = markov, .vocab = vocab };
    return .{ .ctx = holder, .eval = arrayEval };
}

pub const Node = struct {
    /// Stable node ID: assigned in creation order, never reused.
    id: u32,
    /// Parent stable node ID, or `Plan.root_parent` for the root node.
    parent: u32,
    /// Tree depth. Root is depth 0; a child of root is depth 1.
    depth: u32,
    /// Draft token id (vocab index).
    token: u32,
    /// Local conditional log-probability of this token given its parent.
    logp: f32,
    /// Joint path score = parent.path_score + logp. Root is 0.
    path_score: f32,
    /// The round (0-based base-logits position) that created this node.
    round: u32,
};

pub const Plan = struct {
    allocator: std.mem.Allocator,

    /// All nodes created, in stable creation order (id == index in this
    /// slice). Root is index 0.
    nodes: []Node,

    /// Retained node IDs in final display order (path desc, depth asc,
    /// stable id asc; root is rank 0 by construction since its path score
    /// 0 is the global maximum).
    retained: []u32,

    /// Token per retained node, in display order.
    flat_tokens: []u32,

    /// Parent stable node ID per retained node (`root_parent` for root).
    parent_ids: []u32,

    /// Retained rank of the parent per retained node (`no_rank` for root).
    parent_rank: []usize,

    /// Tree depth (== draft position) per retained node.
    depths: []u32,

    /// Local logp per retained node.
    logps: []f32,

    /// Path score per retained node.
    path_scores: []f32,

    /// ancestor_mask[child_row * n_retained + ancestor_row] == true iff
    /// ancestor_row is an ancestor of child_row or equals child_row.
    ancestor_mask: []bool,

    /// Root-to-leaf retrieve rows, one per retained leaf, in display order.
    /// Each row is a list of retained ranks from root to that leaf.
    retrieve_rows: []const []usize,

    /// Winner path: retained ranks of the best root-to-leaf row (the first
    /// retained leaf in display order). The k=1 degenerate case is the
    /// serial chain itself — this is the serial branch oracle's handle.
    winner_path: []usize,

    /// Position of each retained rank in `winner_path`, or `no_rank`.
    winner_pos: []usize,

    /// Total nodes created before the retain cut (== nodes.len).
    node_count: usize,

    /// Retained node count.
    n_retained: usize,

    pub const no_rank: usize = std.math.maxInt(usize);
    pub const root_parent: u32 = std.math.maxInt(u32);

    pub fn deinit(self: *const Plan) void {
        const a = self.allocator;
        a.free(self.nodes);
        a.free(self.retained);
        a.free(self.flat_tokens);
        a.free(self.parent_ids);
        a.free(self.parent_rank);
        a.free(self.depths);
        a.free(self.logps);
        a.free(self.path_scores);
        a.free(self.ancestor_mask);
        for (self.retrieve_rows) |row| a.free(row);
        a.free(self.retrieve_rows);
        a.free(self.winner_path);
        a.free(self.winner_pos);
    }
};

const LocalSortContext = struct {
    base: []const f32,
    markov: []const f32,
};

/// The exact host selector used by DSpark's serial Markov loop.  Do not
/// sanitize NaN here: if row[0] is NaN, every `value > row[0]` comparison is
/// false and the runtime returns zero.  That unusual behavior is part of the
/// k=1 compatibility contract.
pub fn serialArgmax(row: []const f32) u32 {
    std.debug.assert(row.len > 0);
    var best: usize = 0;
    for (row, 0..) |value, index| {
        if (value > row[best]) best = index;
    }
    return @intCast(best);
}

fn rawRowHasNonFinite(base: []const f32, markov_buf: []const f32) bool {
    for (base, markov_buf) |x, bias| {
        if (!std.math.isFinite(x + bias)) return true;
    }
    return false;
}

fn localOrderLess(base: []const f32, markov_buf: []const f32, lhs: u32, rhs: u32) bool {
    const lhs_z = base[lhs] + markov_buf[lhs];
    const rhs_z = base[rhs] + markov_buf[rhs];
    if (lhs_z != rhs_z) return lhs_z > rhs_z;
    return lhs < rhs;
}

fn layerNodeLess(nodes: []const Node, lhs: u32, rhs: u32) bool {
    const lhs_node = nodes[lhs];
    const rhs_node = nodes[rhs];
    if (lhs_node.path_score != rhs_node.path_score) return lhs_node.path_score > rhs_node.path_score;
    return lhs < rhs;
}

/// Plan the PCTree. `base_logits` is row-major [b][vocab] shared-trunk-head
/// logits; `markov` produces the bigram bias row per token (see MarkovScorer).
pub fn plan(
    allocator: std.mem.Allocator,
    b: usize,
    k: usize,
    n: usize,
    vocab: usize,
    base_logits: []const f32,
    markov: MarkovScorer,
    root_token: u32,
) anyerror!Plan {
    if (b < 1 or k < 1 or n < 1 or vocab < 1) return error.InvalidBudget;
    if (b > std.math.maxInt(u32) or vocab > std.math.maxInt(u32)) return error.NodeBudgetExceeded;
    const expected_logits = std.math.mul(usize, b, vocab) catch return error.NodeBudgetExceeded;
    if (base_logits.len != expected_logits) return error.InvalidBaseLogits;
    if (root_token >= vocab) return error.InvalidRootToken;
    // Paper Algorithm 1 keeps every local top-k child in the candidate pool,
    // then limits only the *next* expansion frontier.  Its exact structural
    // maximum is 1 + k + (b - 1) k^2 (root included), not 1 + b*k.
    const k_squared = std.math.mul(usize, k, k) catch return error.NodeBudgetExceeded;
    const tail = if (b > 1)
        std.math.mul(usize, b - 1, k_squared) catch return error.NodeBudgetExceeded
    else
        0;
    const first_layer = std.math.add(usize, 1, k) catch return error.NodeBudgetExceeded;
    const max_nodes = std.math.add(usize, first_layer, tail) catch return error.NodeBudgetExceeded;
    if (max_nodes > @as(usize, std.math.maxInt(u32)) + 1) return error.NodeBudgetExceeded;

    var nodes: std.ArrayListUnmanaged(Node) = .empty;
    defer nodes.deinit(allocator);
    try nodes.append(allocator, .{
        .id = 0,
        .parent = Plan.root_parent,
        .depth = 0,
        .token = root_token,
        .logp = 0.0,
        .path_score = 0.0,
        .round = 0,
    });

    var frontier: std.ArrayListUnmanaged(u32) = .empty;
    defer frontier.deinit(allocator);
    try frontier.append(allocator, 0);

    const markov_buf = try allocator.alloc(f32, vocab);
    defer allocator.free(markov_buf);
    const local_order = try allocator.alloc(u32, vocab);
    defer allocator.free(local_order);

    for (0..b) |round_usize| {
        const round: u32 = @intCast(round_usize);
        const depth: u32 = round + 1;
        const base = base_logits[@as(usize, round) * vocab ..][0..vocab];
        var layer: std.ArrayListUnmanaged(u32) = .empty;
        defer layer.deinit(allocator);

        for (frontier.items) |parent_id| {
            const parent = nodes.items[parent_id];
            try markov.eval(markov.ctx, parent.token, markov_buf);
            if (k > 1 and rawRowHasNonFinite(base, markov_buf)) return error.NonFiniteBranchLogit;
            const lse = rowLogSumExp(base, markov_buf);
            const keep_local = @min(k, vocab);
            if (k == 1) {
                // Must remain raw-logit selection; logp normalization can
                // round two distinct f32 logits to the same value.
                local_order[0] = serialArgmaxAdd(base, markov_buf);
            } else {
                for (local_order, 0..) |*slot, index| slot.* = @intCast(index);
                std.mem.sort(u32, local_order, LocalSortContext{ .base = base, .markov = markov_buf }, struct {
                    fn lt(ctx: LocalSortContext, lhs: u32, rhs: u32) bool {
                        return localOrderLess(ctx.base, ctx.markov, lhs, rhs);
                    }
                }.lt);
            }
            for (local_order[0..keep_local]) |token| {
                const z = base[token] + markov_buf[token];
                const logp: f32 = if (std.math.isFinite(z) and std.math.isFinite(lse))
                    z - lse
                else
                    -std.math.inf(f32);
                if (nodes.items.len >= max_nodes) return error.NodeBudgetExceeded;
                const new_id: u32 = @intCast(nodes.items.len);
                try nodes.append(allocator, .{
                    .id = new_id,
                    .parent = parent_id,
                    .depth = depth,
                    .token = token,
                    .logp = logp,
                    .path_score = parent.path_score + logp,
                    .round = round,
                });
                try layer.append(allocator, new_id);
            }
        }

        // The paper keeps every node above in the global candidate pool;
        // only this expansion handle is width-k.  Stable ID resolves equal
        // path scores without changing the global top-N retention contract.
        std.mem.sort(u32, layer.items, nodes.items, struct {
            fn lt(ctx: []const Node, lhs: u32, rhs: u32) bool {
                return layerNodeLess(ctx, lhs, rhs);
            }
        }.lt);
        var next_frontier: std.ArrayListUnmanaged(u32) = .empty;
        errdefer next_frontier.deinit(allocator);
        try next_frontier.appendSlice(allocator, layer.items[0..@min(k, layer.items.len)]);
        frontier.deinit(allocator);
        frontier = next_frontier;
    }

    const node_count = nodes.items.len;
    return buildPlan(allocator, nodes.items, node_count, n);
}

fn serialArgmaxAdd(base: []const f32, markov_buf: []const f32) u32 {
    std.debug.assert(base.len == markov_buf.len and base.len > 0);
    var best: usize = 0;
    for (base, markov_buf, 0..) |value, bias, index| {
        if (value + bias > base[best] + markov_buf[best]) best = index;
    }
    return @intCast(best);
}

/// log-sum-exp of (base[j] + markov_buf[j]) in f64, ascending j order.
/// NaN entries are skipped (they clamp to -inf at the logp level).
fn rowLogSumExp(base: []const f32, markov_buf: []const f32) f32 {
    var maxv: f32 = -std.math.inf(f32);
    var has_real = false;
    for (base, markov_buf) |x, mj| {
        const z = x + mj;
        if (!std.math.isNan(z)) {
            maxv = @max(maxv, z);
            has_real = true;
        }
    }
    if (!has_real) return -std.math.inf(f32);
    if (maxv == std.math.inf(f32)) return std.math.inf(f32);
    if (maxv == -std.math.inf(f32)) return -std.math.inf(f32); // all -inf row
    var sum: f64 = 0.0;
    for (base, markov_buf) |x, mj| {
        const z = x + mj;
        if (std.math.isNan(z)) continue;
        sum += @exp(@as(f64, z) - @as(f64, maxv));
    }
    if (sum == 0.0 or std.math.isInf(sum)) return maxv;
    return @floatCast(@log(sum) + @as(f64, maxv));
}

fn buildPlan(
    allocator: std.mem.Allocator,
    nodes_in: []const Node,
    node_count: usize,
    n: usize,
) std.mem.Allocator.Error!Plan {
    // Sorted node ids by (path desc, depth asc, id asc). Root always leads:
    // path 0 is the maximum and ties break on depth (0 is the minimum).
    const order = try allocator.alloc(u32, node_count);
    defer allocator.free(order);
    for (order, 0..) |*o, idx| o.* = @intCast(idx);
    std.mem.sort(u32, order, nodes_in, struct {
        fn lt(ctx: []const Node, a: u32, b: u32) bool {
            const na = ctx[a];
            const nb = ctx[b];
            if (na.path_score != nb.path_score) return na.path_score > nb.path_score;
            if (na.depth != nb.depth) return na.depth < nb.depth;
            return a < b;
        }
    }.lt);

    // Retain in display order while admitting each missing parent chain as one
    // unit. Normal planner nodes are score-monotone, so this selects the same
    // top-N nodes as a plain prefix cut. The explicit admission test is still
    // necessary: it keeps `n` strict even if a future score/key change (or a
    // direct internal caller) presents a child ahead of an unretained parent.
    const keep = @min(n, node_count);
    const retained_bit = try allocator.alloc(bool, node_count);
    defer allocator.free(retained_bit);
    @memset(retained_bit, false);

    // `plan` always creates root id 0 and `n >= 1`; retaining it first makes
    // root presence and the cap independent of score ordering.
    retained_bit[0] = true;
    var retained_count: usize = 1;
    for (order) |id| {
        if (retained_bit[id]) continue;

        var missing: usize = 0;
        var cur = id;
        while (!retained_bit[cur]) {
            missing += 1;
            cur = nodes_in[cur].parent;
        }

        if (missing > keep - retained_count) continue;
        cur = id;
        while (!retained_bit[cur]) {
            retained_bit[cur] = true;
            retained_count += 1;
            cur = nodes_in[cur].parent;
        }
    }
    std.debug.assert(retained_count <= keep);
    const n_retained = retained_count;

    const retained = try allocator.alloc(u32, n_retained);
    errdefer allocator.free(retained);
    var w: usize = 0;
    for (order) |id| {
        if (retained_bit[id]) {
            retained[w] = id;
            w += 1;
        }
    }
    std.debug.assert(w == n_retained);

    // rank_of[node_id] -> retained rank (or no_rank).
    const rank_of = try allocator.alloc(usize, node_count);
    defer allocator.free(rank_of);
    @memset(rank_of, Plan.no_rank);
    for (retained, 0..) |id, r| rank_of[id] = r;

    const flat_tokens = try allocator.alloc(u32, n_retained);
    errdefer allocator.free(flat_tokens);
    const parent_ids = try allocator.alloc(u32, n_retained);
    errdefer allocator.free(parent_ids);
    const parent_rank = try allocator.alloc(usize, n_retained);
    errdefer allocator.free(parent_rank);
    const depths = try allocator.alloc(u32, n_retained);
    errdefer allocator.free(depths);
    const logps = try allocator.alloc(f32, n_retained);
    errdefer allocator.free(logps);
    const path_scores = try allocator.alloc(f32, n_retained);
    errdefer allocator.free(path_scores);
    for (retained, 0..) |id, r| {
        const node = nodes_in[id];
        flat_tokens[r] = node.token;
        parent_ids[r] = node.parent;
        parent_rank[r] = if (node.parent == Plan.root_parent) Plan.no_rank else rank_of[node.parent];
        depths[r] = node.depth;
        logps[r] = node.logp;
        path_scores[r] = node.path_score;
    }

    // Leaf detection within the retained subgraph.
    const has_child = try allocator.alloc(bool, n_retained);
    defer allocator.free(has_child);
    @memset(has_child, false);
    for (parent_rank) |pr| {
        if (pr != Plan.no_rank) has_child[pr] = true;
    }

    // Retrieve rows: root-to-leaf per retained leaf, in display order.
    var leaf_ranks: std.ArrayListUnmanaged(usize) = .empty;
    defer leaf_ranks.deinit(allocator);
    for (0..n_retained) |r| {
        if (!has_child[r]) try leaf_ranks.append(allocator, r);
    }
    const retrieve_rows = try allocator.alloc([]usize, leaf_ranks.items.len);
    var retrieve_rows_initialized: usize = 0;
    errdefer {
        for (retrieve_rows[0..retrieve_rows_initialized]) |row| allocator.free(row);
        allocator.free(retrieve_rows);
    }
    for (leaf_ranks.items, 0..) |leaf, ri| {
        const row_len = std.math.add(usize, @as(usize, depths[leaf]), 1) catch return error.OutOfMemory;
        const row = try allocator.alloc(usize, row_len);
        var idx: usize = leaf;
        var pos: usize = row.len;
        while (idx != Plan.no_rank) {
            pos -= 1;
            row[pos] = idx;
            idx = parent_rank[idx];
        }
        std.debug.assert(pos == 0);
        retrieve_rows[ri] = row;
        retrieve_rows_initialized += 1;
    }

    // Winner: the first row (its leaf leads the display order).
    const winner_path = try allocator.alloc(usize, retrieve_rows[0].len);
    errdefer allocator.free(winner_path);
    @memcpy(winner_path, retrieve_rows[0]);

    const winner_pos = try allocator.alloc(usize, n_retained);
    errdefer allocator.free(winner_pos);
    @memset(winner_pos, Plan.no_rank);
    for (winner_path, 0..) |r, p| winner_pos[r] = p;

    const ancestor_mask_len = std.math.mul(usize, n_retained, n_retained) catch return error.OutOfMemory;
    const ancestor_mask = try allocator.alloc(bool, ancestor_mask_len);
    errdefer allocator.free(ancestor_mask);
    @memset(ancestor_mask, false);
    for (retained, 0..) |id, child_rank| {
        var cur: u32 = id;
        while (true) {
            const ar = rank_of[cur];
            if (ar == Plan.no_rank) break;
            ancestor_mask[child_rank * n_retained + ar] = true;
            if (nodes_in[cur].parent == Plan.root_parent) break;
            cur = nodes_in[cur].parent;
        }
    }

    const owned_nodes = try allocator.dupe(Node, nodes_in);
    errdefer allocator.free(owned_nodes);
    return .{
        .allocator = allocator,
        .nodes = owned_nodes,
        .retained = retained,
        .flat_tokens = flat_tokens,
        .parent_ids = parent_ids,
        .parent_rank = parent_rank,
        .depths = depths,
        .logps = logps,
        .path_scores = path_scores,
        .ancestor_mask = ancestor_mask,
        .retrieve_rows = retrieve_rows,
        .winner_path = winner_path,
        .winner_pos = winner_pos,
        .node_count = node_count,
        .n_retained = n_retained,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn naiveChain(allocator: std.mem.Allocator, b: usize, vocab: usize, base: []const f32, markov: []const f32, root: u32) ![]u32 {
    const out = try allocator.alloc(u32, b);
    var prev = root;
    for (0..b) |d| {
        var best: usize = 0;
        var best_z: f32 = -std.math.inf(f32);
        const brow = base[d * vocab ..][0..vocab];
        const mrow = markov[@as(usize, prev) * vocab ..][0..vocab];
        for (0..vocab) |j| {
            const z = brow[j] + mrow[j];
            if (z > best_z) {
                best_z = z;
                best = j;
            }
        }
        out[d] = @intCast(best);
        prev = @intCast(best);
    }
    return out;
}

const prng_seed = 0x5EED_47EE;

test "pctree: k=1 chain equals the naive greedy reference chain" {
    const allocator = testing.allocator;
    var rng = std.Random.DefaultPrng.init(prng_seed);
    var markov: [64]f32 = undefined;
    var base_buf: [16 * 8]f32 = undefined;

    for (0..40) |_| {
        const b = 1 + rng.random().intRangeAtMost(usize, 0, 15);
        const vocab: usize = 8;
        for (&base_buf) |*v| v.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -10, 10)));
        for (&markov) |*v| v.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -10, 10)));
        const root = rng.random().intRangeAtMost(u32, 0, 7);

        var holder: ArrayScorerCtx = undefined;
        const scorer = newArrayScorer(&markov, vocab, &holder);
        var p = try plan(allocator, b, 1, b + 1, vocab, base_buf[0 .. b * vocab], scorer, root);
        defer p.deinit();
        const ref = try naiveChain(allocator, b, vocab, base_buf[0 .. b * vocab], &markov, root);
        defer allocator.free(ref);

        try testing.expectEqual(b + 1, p.flat_tokens.len);
        try testing.expectEqual(root, p.flat_tokens[0]);
        try testing.expectEqualSlices(u32, ref, p.flat_tokens[1..]);
        // k=1: the winner path is the whole chain, ranks 0..b
        try testing.expectEqual(b + 1, p.winner_path.len);
        for (p.winner_path, 0..) |r, i| try testing.expectEqual(i, r);
        // ancestor mask: rank r's ancestors are exactly ranks 0..r (chain)
        for (0..p.n_retained) |c| {
            for (0..p.n_retained) |a| {
                try testing.expectEqual(c >= a, p.ancestor_mask[c * p.n_retained + a]);
            }
        }
    }
}

test "pctree: deterministic tie-break — equal scores pick the lowest token id" {
    const allocator = testing.allocator;
    const vocab: usize = 4;
    // base row: every z ties (all-zero markov) -> token 0 must win.
    const base = [_]f32{ 0, 0, 0, 0 };
    const markov = @as([16]f32, @splat(0.0));
    var holder: ArrayScorerCtx = undefined;
    const scorer = newArrayScorer(&markov, vocab, &holder);
    var p1 = try plan(allocator, 1, 1, 2, vocab, &base, scorer, 0);
    defer p1.deinit();
    var p2 = try plan(allocator, 1, 1, 2, vocab, &base, scorer, 0);
    defer p2.deinit();
    try testing.expectEqual(@as(u32, 0), p1.flat_tokens[1]);
    try testing.expectEqualSlices(u32, p1.retained, p2.retained);
    try testing.expectEqualSlices(u32, p1.flat_tokens, p2.flat_tokens);
    try testing.expectEqualSlices(bool, p1.ancestor_mask, p2.ancestor_mask);
}

test "pctree: mutation-sensitive — perturbing one logit changes the plan" {
    const allocator = testing.allocator;
    const vocab: usize = 4;
    const b: usize = 2;
    var base = [_]f32{ 2, 1.5, -20, -20, 1, 0.5, -20, -20 };
    const markov = @as([16]f32, @splat(0.0));
    var holder: ArrayScorerCtx = undefined;
    const scorer = newArrayScorer(&markov, vocab, &holder);

    var p1 = try plan(allocator, b, 1, b + 1, vocab, &base, scorer, 0);
    defer p1.deinit();
    try testing.expectEqual(@as(u32, 0), p1.flat_tokens[1]);
    try testing.expectEqual(@as(u32, 0), p1.flat_tokens[2]);
    // Drop the winning logit so token 1 wins position 0; the chain must
    // change (mutation-sensitivity) while staying finite.
    base[1] = 2.5;
    var p2 = try plan(allocator, b, 1, b + 1, vocab, &base, scorer, 0);
    defer p2.deinit();
    try testing.expectEqual(@as(u32, 1), p2.flat_tokens[1]);
    try testing.expectEqual(@as(u32, 0), p2.flat_tokens[2]);
    try testing.expect(!std.mem.eql(u32, p1.flat_tokens, p2.flat_tokens));
}

test "pctree: tree structure fixture (hand-verified against a debug run)" {
    const allocator = testing.allocator;
    const vocab: usize = 4;
    const base = [_]f32{
        10, 9, -100, -100, // L[0]: argmax is token 0 (10), then token 1 (9)
        4, -100, -100, 5, // L[1]
    };
    // M[*][*] all zero for the root's row and M[0]; M[1] penalizes tokens
    // 1/2 so token 1's children are {3, 0}.  PCTree keeps both parents'
    // local top-2 children in its candidate pool, then expands only the
    // best two from that completed layer on a later stage.
    const markov = [_]f32{ 0, 0, 0, 0, 0, -20, -20, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var holder: ArrayScorerCtx = undefined;
    const scorer = newArrayScorer(&markov, vocab, &holder);

    var p = try plan(allocator, 2, 2, 4, vocab, &base, scorer, 0);
    defer p.deinit();

    try testing.expectEqual(@as(usize, 7), p.node_count);
    try testing.expectEqual(@as(usize, 4), p.n_retained);

    // ids: 0=root, 1=token0(d1), 2=token1(d1), then the complete local
    // top-2 pool: 3/4 from id1 and 5/6 from id2.  The global retain budget
    // cuts the display tree to the same four highest scored nodes.
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 3, 2 }, p.retained);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 0, 3, 1 }, p.flat_tokens);
    try testing.expectEqualSlices(u32, &[_]u32{ Plan.root_parent, 0, 1, 0 }, p.parent_ids);
    try testing.expectEqualSlices(usize, &[_]usize{ Plan.no_rank, 0, 1, 0 }, p.parent_rank);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2, 1 }, p.depths);

    const logp_top = @as(f32, -0.31326169);
    try testing.expectApproxEqAbs(@as(f32, 0.0), p.path_scores[0], 1e-5);
    try testing.expectApproxEqAbs(logp_top, p.path_scores[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -0.6265234), p.path_scores[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -1.3132617), p.path_scores[3], 1e-4);
    try testing.expectApproxEqAbs(logp_top, p.logps[1], 1e-4);
    try testing.expectApproxEqAbs(logp_top, p.logps[2], 1e-4); // token 3 vs L[1] alone

    // leaves (within the retained subgraph): id3 (rank 2), then id2 (rank 3)
    try testing.expectEqual(@as(usize, 2), p.retrieve_rows.len);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, p.retrieve_rows[0]);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 3 }, p.retrieve_rows[1]);
    // winner: the first leaf's row
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, p.winner_path);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, Plan.no_rank }, p.winner_pos);

    // ancestor mask (child rank x ancestor rank), self included
    const expect_mask = [_]bool{
        true, false, false, false,
        true, true,  false, false,
        true, true,  true,  false,
        true, false, false, true,
    };
    for (expect_mask, 0..) |want, idx| try testing.expectEqual(want, p.ancestor_mask[idx]);
}

test "pctree: strict retain and branch budgets preserve stable ids under ties" {
    const allocator = testing.allocator;
    const b: usize = 4;
    const k: usize = 3;
    const n: usize = 5;
    const vocab: usize = 3;
    // Every local and global score ties. This forces all public tie-breaks:
    // local token order, global token then parent order, and stable creation
    // ids across every branch-width-limited frontier.
    const base = @as([b * vocab]f32, @splat(0.0));
    const markov = @as([vocab * vocab]f32, @splat(0.0));
    var holder: ArrayScorerCtx = undefined;
    const scorer = newArrayScorer(&markov, vocab, &holder);
    var p = try plan(allocator, b, k, n, vocab, &base, scorer, 2);
    defer p.deinit();

    // Algorithm 1 keeps every local top-k child in the candidate pool, while
    // only the next frontier is width-k.  This is 1 + k + (b-1)k^2 nodes.
    try testing.expectEqual(@as(usize, 1) + k + (b - 1) * k * k, p.node_count);
    try testing.expectEqual(n, p.n_retained);
    try testing.expect(p.n_retained <= n);
    for (p.nodes, 0..) |node, id| try testing.expectEqual(@as(u32, @intCast(id)), node.id);
    for (1..b + 1) |depth| {
        var at_depth: usize = 0;
        for (p.nodes) |node| at_depth += @intFromBool(node.depth == depth);
        try testing.expectEqual(if (depth == 1) k else k * k, at_depth);
    }

    // At depth two, every local child stays in the pool.  Creation is parent
    // order then token order, and the k-wide *next* frontier resolves equal
    // scores by stable node id.
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, &[_]u32{ p.nodes[4].token, p.nodes[5].token, p.nodes[6].token });
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 1 }, &[_]u32{ p.nodes[4].parent, p.nodes[5].parent, p.nodes[6].parent });
}

test "pctree: retain cap remains strict under a nonmonotone score negative control" {
    const allocator = testing.allocator;
    // Planner-produced scores are monotone along a path, but buildPlan's
    // defensive closure must not silently exceed n if that precondition is
    // broken by a future score/key change. The old top-N-then-close code kept
    // all three nodes here even though n == 2.
    const nodes = [_]Node{
        .{ .id = 0, .parent = Plan.root_parent, .depth = 0, .token = 0, .logp = 0.0, .path_score = 0.0, .round = 0 },
        .{ .id = 1, .parent = 0, .depth = 1, .token = 1, .logp = -10.0, .path_score = -10.0, .round = 0 },
        .{ .id = 2, .parent = 1, .depth = 2, .token = 2, .logp = 11.0, .path_score = 1.0, .round = 1 },
    };
    var p = try buildPlan(allocator, &nodes, nodes.len, 2);
    defer p.deinit();

    try testing.expectEqual(@as(usize, 2), p.n_retained);
    try testing.expect(p.n_retained <= 2);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1 }, p.retained);
    try testing.expectEqualSlices(usize, &[_]usize{ Plan.no_rank, 0 }, p.parent_rank);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, p.winner_path);
}

test "pctree: invalid inputs are rejected" {
    const allocator = testing.allocator;
    const base = [_]f32{ 1, 1, 1, 1 };
    const markov = @as([16]f32, @splat(0.0));
    var holder: ArrayScorerCtx = undefined;
    const scorer = newArrayScorer(&markov, 4, &holder);

    try testing.expectError(error.InvalidBaseLogits, plan(allocator, 2, 1, 2, 4, &base, scorer, 0)); // len 4 != 2*4
    try testing.expectError(error.InvalidBudget, plan(allocator, 0, 1, 2, 4, &[_]f32{}, scorer, 0)); // b=0
    try testing.expectError(error.InvalidBudget, plan(allocator, 1, 0, 2, 4, &base, scorer, 0)); // k=0
    try testing.expectError(error.InvalidBudget, plan(allocator, 1, 1, 0, 4, &base, scorer, 0)); // n=0
    try testing.expectError(error.InvalidBudget, plan(allocator, 1, 1, 2, 0, &base, scorer, 0)); // vocab=0
    try testing.expectError(error.InvalidRootToken, plan(allocator, 1, 1, 2, 4, &base, scorer, 4)); // root OOB
    try testing.expectError(error.NodeBudgetExceeded, plan(allocator, @as(usize, std.math.maxInt(u32)) + 1, 1, 2, 1, &.{}, scorer, 0));
    try testing.expectError(error.NodeBudgetExceeded, plan(allocator, 1, 1, 2, @as(usize, std.math.maxInt(u32)) + 1, &.{}, scorer, 0));

    var short_holder: ArrayScorerCtx = undefined;
    const short_scorer = newArrayScorer(markov[0..3], 4, &short_holder);
    try testing.expectError(error.InvalidMarkovScorer, plan(allocator, 1, 1, 2, 4, &base, short_scorer, 0));
}

test "pctree: every planner allocation failure is leak-free" {
    const base = [_]f32{
        4, 3, 2, 1,
        1, 4, 3, 2,
        2, 1, 4, 3,
    };
    const markov = @as([16]f32, @splat(0.0));
    try testing.checkAllAllocationFailures(testing.allocator, struct {
        fn f(allocator: std.mem.Allocator) !void {
            var holder: ArrayScorerCtx = undefined;
            const scorer = newArrayScorer(&markov, 4, &holder);
            var p = try plan(allocator, 3, 3, 10, 4, &base, scorer, 0);
            defer p.deinit();
            try testing.expect(p.retrieve_rows.len > 0);
        }
    }.f, .{});
}

test "pctree: retain cap larger than the tree keeps everything" {
    const allocator = testing.allocator;
    const vocab: usize = 4;
    const base = [_]f32{ 5, 4, 3, 2, 1, 0, -1, -2 };
    const markov = @as([16]f32, @splat(0.0));
    var holder: ArrayScorerCtx = undefined;
    const scorer = newArrayScorer(&markov, vocab, &holder);
    var p = try plan(allocator, 2, 3, 100, vocab, &base, scorer, 0);
    defer p.deinit();
    try testing.expectEqual(p.node_count, p.n_retained);
    for (p.winner_path, 0..) |r, i| try testing.expectEqual(i, p.winner_pos[r]);
    const last = p.retrieve_rows[0][p.retrieve_rows[0].len - 1];
    try testing.expectEqual(p.depths[last] + 1, p.retrieve_rows[0].len);
}

test "pctree: k=1 raw logits preserve one-ULP order before log-softmax" {
    const allocator = testing.allocator;
    const vocab: usize = 2;
    // These values differ by exactly one f32 ULP.  The old planner ranked
    // rounded f32 log-probabilities and collapsed this difference to a tie,
    // choosing token zero.  DSpark ranks raw logits and chooses token one.
    const one = @as(f32, @bitCast(@as(u32, 0x3f80_0000)));
    const one_ulp_up = @as(f32, @bitCast(@as(u32, 0x3f80_0001)));
    const base = [_]f32{ one, one_ulp_up };
    const markov = @as([4]f32, @splat(0.0));
    var holder: ArrayScorerCtx = undefined;
    const scorer = newArrayScorer(&markov, vocab, &holder);
    var p = try plan(allocator, 1, 1, 2, vocab, &base, scorer, 0);
    defer p.deinit();
    try testing.expectEqual(@as(u32, 1), p.flat_tokens[1]);
}

test "pctree: k=1 follows serial NaN and infinity behavior exactly" {
    const allocator = testing.allocator;
    const vocab: usize = 4;
    const markov = @as([16]f32, @splat(0.0));
    var holder: ArrayScorerCtx = undefined;
    const scorer = newArrayScorer(&markov, vocab, &holder);

    // Runtime starts from index zero.  A NaN there wins by poisoning every
    // strict-greater comparison, even though later values are finite.
    const nan_first = [_]f32{ std.math.nan(f32), 9, 8, 7 };
    var first = try plan(allocator, 1, 1, 2, vocab, &nan_first, scorer, 0);
    defer first.deinit();
    try testing.expectEqual(@as(u32, 0), first.flat_tokens[1]);

    const nan_later = [_]f32{ 1, std.math.nan(f32), 2, 0 };
    var later = try plan(allocator, 1, 1, 2, vocab, &nan_later, scorer, 0);
    defer later.deinit();
    try testing.expectEqual(@as(u32, 2), later.flat_tokens[1]);

    const plus_inf = [_]f32{ 1, std.math.inf(f32), std.math.inf(f32), 0 };
    var inf = try plan(allocator, 1, 1, 2, vocab, &plus_inf, scorer, 0);
    defer inf.deinit();
    try testing.expectEqual(@as(u32, 1), inf.flat_tokens[1]);
    for (inf.logps, inf.path_scores) |lp, ps| {
        try testing.expect(!std.math.isNan(lp));
        try testing.expect(!std.math.isNan(ps));
    }

    try testing.expectError(error.NonFiniteBranchLogit, plan(allocator, 1, 2, 3, vocab, &nan_later, scorer, 0));
}

test "pctree: random tree sweep — invariants hold and runs are deterministic" {
    const allocator = testing.allocator;
    var rng = std.Random.DefaultPrng.init(prng_seed + 1);

    for (0..60) |_| {
        const b: usize = 1 + rng.random().intRangeAtMost(usize, 0, 7);
        const vocab: usize = 4 + rng.random().intRangeAtMost(usize, 0, 7);
        const k: usize = 1 + rng.random().intRangeAtMost(usize, 0, 3);
        const n: usize = 1 + rng.random().intRangeAtMost(usize, 0, 23);
        const base = try allocator.alloc(f32, b * vocab);
        defer allocator.free(base);
        const markov = try allocator.alloc(f32, vocab * vocab);
        defer allocator.free(markov);
        for (base) |*v| v.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -8, 8)));
        for (markov) |*v| v.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -8, 8)));
        const root = rng.random().intRangeAtMost(u32, 0, @intCast(vocab - 1));

        var holder: ArrayScorerCtx = undefined;
        const scorer = newArrayScorer(markov, vocab, &holder);
        var p1 = try plan(allocator, b, k, n, vocab, base, scorer, root);
        defer p1.deinit();
        var p2 = try plan(allocator, b, k, n, vocab, base, scorer, root);
        defer p2.deinit();

        // determinism: byte-identical outputs
        try testing.expectEqualSlices(u32, p1.retained, p2.retained);
        try testing.expectEqualSlices(u32, p1.flat_tokens, p2.flat_tokens);
        try testing.expectEqualSlices(bool, p1.ancestor_mask, p2.ancestor_mask);
        try testing.expectEqualSlices(f32, p1.path_scores, p2.path_scores);
        try testing.expectEqualSlices(usize, p1.winner_path, p2.winner_path);
        try testing.expectEqual(p1.n_retained, p2.n_retained);
        try testing.expect(p1.n_retained <= n);

        // prefix-closed: every retained non-root's parent is retained
        for (0..p1.n_retained) |r| {
            if (p1.parent_ids[r] != Plan.root_parent) {
                try testing.expect(p1.parent_rank[r] != Plan.no_rank);
                try testing.expectEqual(p1.parent_ids[r], p1.nodes[p1.retained[p1.parent_rank[r]]].id);
            }
        }
        // root is rank 0
        try testing.expectEqual(@as(u32, 0), p1.retained[0]);
        try testing.expectEqual(root, p1.flat_tokens[0]);
        try testing.expectEqual(@as(f32, 0.0), p1.path_scores[0]);

        // retrieve rows are root-to-leaf and rank-consistent
        for (p1.retrieve_rows) |row| {
            try testing.expectEqual(@as(usize, 0), row[0]);
            try testing.expectEqual(row.len - 1, row.len - 1);
            const leaf = row[row.len - 1];
            try testing.expectEqual(p1.depths[leaf] + 1, row.len);
            for (row[1..], 0..) |_, ri_off| {
                const ri = ri_off + 1;
                try testing.expectEqual(row[ri - 1], p1.parent_rank[row[ri]]);
            }
        }
        // winner_pos consistent with the winner row
        for (p1.winner_path, 0..) |r, i| try testing.expectEqual(i, p1.winner_pos[r]);

        // ancestors match the mask exactly
        for (0..p1.n_retained) |c| {
            var chain: std.ArrayListUnmanaged(usize) = .empty;
            defer chain.deinit(allocator);
            var cur: usize = c;
            while (cur != Plan.no_rank) {
                try chain.append(allocator, cur);
                cur = p1.parent_rank[cur];
            }
            for (0..p1.n_retained) |a| {
                try testing.expectEqual(contains(chain.items, a), p1.ancestor_mask[c * p1.n_retained + a]);
            }
        }
        // no NaN / no inf anywhere
        for (p1.logps, p1.path_scores) |lp, ps| {
            try testing.expect(!std.math.isNan(lp) and !std.math.isInf(lp));
            try testing.expect(!std.math.isNan(ps) and !std.math.isInf(ps));
        }
    }
}

fn contains(haystack: []const usize, needle: usize) bool {
    for (haystack) |h| {
        if (h == needle) return true;
    }
    return false;
}

test "pctree: all -inf k=1 row matches serial and never creates NaN" {
    const allocator = testing.allocator;
    const vocab: usize = 3;
    const base = [_]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    const markov = @as([9]f32, @splat(0.0));
    var holder: ArrayScorerCtx = undefined;
    const scorer = newArrayScorer(&markov, vocab, &holder);
    var p = try plan(allocator, 1, 1, 2, vocab, &base, scorer, 0);
    defer p.deinit();
    try testing.expectEqual(@as(u32, 0), p.flat_tokens[1]);
    try testing.expect(!std.math.isNan(p.logps[1]));
    try testing.expect(!std.math.isNan(p.path_scores[1]));
    try testing.expect(std.math.isInf(p.logps[1]));
}
