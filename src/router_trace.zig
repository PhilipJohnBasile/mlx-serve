//! Per-request DeepSeek router trace, bounded binary sink owned by the
//! scheduler `Slot`. Exposed to routing through `ForwardCtx.router_trace`.
//!
//! Design (approved):
//!   - No model-global state; one optional sink per Slot.
//!   - Fixed 24-byte little-endian records (not ABI structs).
//!   - Capacity preallocated (hard ceiling 64 MiB); overflow STOPS recording
//!     and never fails the request.
//!   - seal() captures metadata before markFinished; flushOnce() writes
//!     `<dir>/<req>.<ext>.tmp` then atomically renames to final. Idempotent.
//!   - Trace-run timing is NOT performance evidence. GPU routing is captured
//!     in deepseek_v4.zig (evals + copies inds to host) then this module's
//!     appendHost is used; this module is std-only (no mlx types).

const std = @import("std");
const builtin = @import("builtin");

pub const Phase = enum(u8) {
    prefill = 0,
    decode = 1,
    verify = 2,

    pub inline fn toU8(self: Phase) u8 {
        return @intFromEnum(self);
    }
};

const RECORD_LEN = 24; // u64 req + u32 pos + u32 tok + u8 layer + u8 phase + u8[6]
const MAX_CAPACITY = 64 * 1024 * 1024; // 64 MiB hard ceiling

fn storeLE(dst: []u8, comptime T: type, value: T) void {
    std.debug.assert(dst.len == @sizeOf(T));
    const bytes = std.mem.toBytes(value);
    if (builtin.cpu.arch.endian() == .big) {
        for (bytes, 0..) |b, i| dst[bytes.len - 1 - i] = b;
    } else {
        for (bytes, 0..) |b, i| dst[i] = b;
    }
}

pub const Meta = struct {
    finish_reason: []const u8 = "",
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    error_name: ?[]const u8 = null,
};

pub const RouterTraceSink = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    request_id: u64,
    output_dir: []const u8, // directory (relative or absolute); borrowed
    records: std.ArrayList(u8),
    capacity: usize,
    overflowed: bool = false,
    malformed: bool = false,
    sealed: bool = false,
    flushed: bool = false,
    meta: Meta = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        request_id: u64,
        output_dir: []const u8,
        estimated_records: usize,
    ) !RouterTraceSink {
        const est = @as(usize, @max(estimated_records, 1));
        const cap = @min((est * RECORD_LEN) + 1, MAX_CAPACITY);
        const records = try std.ArrayList(u8).initCapacity(allocator, cap);
        return .{
            .allocator = allocator,
            .io = io,
            .request_id = request_id,
            .output_dir = output_dir,
            .records = records,
            .capacity = cap,
        };
    }

    pub fn deinit(self: *RouterTraceSink) void {
        self.records.deinit(self.allocator);
    }

    fn append16(self: *RouterTraceSink, rec: [RECORD_LEN]u8) void {
        if (self.overflowed or self.sealed) return;
        if (self.records.items.len + RECORD_LEN > self.capacity) {
            self.overflowed = true;
            return;
        }
        self.records.appendSliceAssumeCapacity(&rec);
    }

    pub fn appendHost(
        self: *RouterTraceSink,
        phase: Phase,
        token_base: u32,
        layer_id: u32,
        ids_tokens: []const u32,
        indices: []const i32,
        seq: usize,
        k: usize,
    ) void {
        if (self.overflowed or self.malformed or self.sealed) return;

        if (k == 0 or
            k > 6 or
            layer_id > 255 or
            seq > ids_tokens.len or
            seq > indices.len / k)
        {
            self.malformed = true;
            return;
        }
        var rec: [RECORD_LEN]u8 = undefined;
        for (0..seq) |t| {
            storeLE(rec[0..8], u64, self.request_id);
            storeLE(rec[8..12], u32, token_base + @as(u32, @intCast(t)));
            storeLE(rec[12..16], u32, ids_tokens[t]);
            rec[16] = @intCast(layer_id & 0xff);
            rec[17] = phase.toU8();
            var j: usize = 0;
            while (j < 6) : (j += 1) {
                if (j < k) {
                    const e: i32 = indices[t * k + j];
                    if (e < 0 or e > 255) {
                        self.malformed = true;
                        return;
                    }
                    rec[18 + j] = @intCast(@as(u32, @bitCast(e)) & 0xff);
                } else {
                    rec[18 + j] = 0xff;
                }
            }
            self.append16(rec);
        }
    }

    pub fn seal(self: *RouterTraceSink, metaFNS: Meta) void {
        const m = metaFNS;
        self.meta = m;
        self.sealed = true;
    }

    fn joinName(buf: *[160]u8, req: u64, suffix: []const u8) ?[]const u8 {
        return std.fmt.bufPrint(buf, "{d}{s}", .{ req, suffix }) catch null;
    }

    pub fn flushOnce(self: *RouterTraceSink) !void {
        if (self.flushed) return;
        const cwd = std.Io.Dir.cwd();
        var dir = try cwd.openDir(self.io, self.output_dir, .{});
        defer dir.close(self.io);

        var bin_tmp: [160]u8 = undefined;
        var bin_fin: [160]u8 = undefined;
        var json_tmp: [160]u8 = undefined;
        var json_fin: [160]u8 = undefined;
        const bt = joinName(&bin_tmp, self.request_id, ".bin.tmp") orelse return error.NameTooLong;
        const bf = joinName(&bin_fin, self.request_id, ".bin") orelse return error.NameTooLong;
        const jt = joinName(&json_tmp, self.request_id, ".json.tmp") orelse return error.NameTooLong;
        const jf = joinName(&json_fin, self.request_id, ".json") orelse return error.NameTooLong;

        {
            var f = try dir.createFile(self.io, bt, .{});
            var wb: [8192]u8 = undefined;
            var fw = f.writer(self.io, &wb);
            try fw.interface.writeAll(self.records.items);
            try fw.interface.flush();
            f.close(self.io);
        }
        {
            var jf2 = try dir.createFile(self.io, jt, .{});
            var jb: [2048]u8 = undefined;
            var jfw = jf2.writer(self.io, &jb);
            const m = self.meta;
            try jfw.interface.print(
                "{{\"request_id\":{d},\"finish_reason\":\"{s}\",\"prompt_tokens\":{d}," ++
                    "\"completion_tokens\":{d},\"overflowed\":{any},\"malformed\":{any}," ++
                    "\"records\":{d},\"bytes\":{d},\"error_name\":\"{s}\"}}",
                .{
                    self.request_id,
                    m.finish_reason,
                    m.prompt_tokens,
                    m.completion_tokens,
                    self.overflowed,
                    self.malformed,
                    self.records.items.len / RECORD_LEN,
                    self.records.items.len,
                    if (m.error_name) |en| en else "",
                },
            );
            try jfw.interface.flush();
            jf2.close(self.io);
        }
        try dir.rename(bt, dir, bf, self.io);
        try dir.rename(jt, dir, jf, self.io);
        self.flushed = true;
    }
};

pub const Call = struct {
    sink: *RouterTraceSink,
    phase: Phase,
    token_base: u32,
};

// ── unit tests (TDD-first) ────────────────────────────────────────────────

const testing = std.testing;

test "router_trace: 24-byte record encode fields (little-endian)" {
    const a = testing.allocator;
    var sink = try RouterTraceSink.init(a, testing.io, 1, ".", 3);
    defer sink.deinit();
    const ids = [_]u32{ 7, 8, 9 };
    const indices = [_]i32{ 11, 22, 33, 44, 55, 66, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    sink.appendHost(.prefill, 100, 5, &ids, &indices, ids.len, 6);
    try testing.expectEqual(@as(usize, 72), sink.records.items.len);
    try testing.expectEqual(@as(u8, 1), sink.records.items[0]);
    try testing.expectEqual(@as(u8, 100), sink.records.items[8]);
    try testing.expectEqual(@as(u8, 7), sink.records.items[12]);
    try testing.expectEqual(@as(u8, 5), sink.records.items[16]);
    try testing.expectEqual(@as(u8, 0), sink.records.items[17]);
    try testing.expectEqual(@as(u8, 11), sink.records.items[18]);
    try testing.expectEqual(@as(u8, 66), sink.records.items[23]);
}

test "router_trace: padded k<6 fills 0xff; overflow stops recording" {
    const a = testing.allocator;
    var sink = try RouterTraceSink.init(a, testing.io, 2, ".", 2);
    defer sink.deinit();
    const ids = [_]u32{ 1, 2 };
    const indices = [_]i32{ 1, 2, 3, 4 };
    sink.appendHost(.decode, 0, 3, &ids, &indices, 1, 4);
    try testing.expectEqual(@as(u8, 1), sink.records.items[18]);
    try testing.expectEqual(@as(u8, 4), sink.records.items[21]);
    try testing.expectEqual(@as(u8, 0xff), sink.records.items[22]);
    try testing.expectEqual(@as(u8, 0xff), sink.records.items[23]);
    sink.appendHost(.decode, 0, 3, &ids, &indices, 1, 4);
    sink.appendHost(.decode, 0, 4, &ids, &indices, 1, 4);
    try testing.expect(sink.overflowed);
}

test "router_trace: seal then append is ignored" {
    const a = testing.allocator;
    var sink = try RouterTraceSink.init(a, testing.io, 3, ".", 4);
    defer sink.deinit();
    sink.seal(.{ .finish_reason = "length", .prompt_tokens = 8, .completion_tokens = 5 });
    const ids = [_]u32{ 1, 2 };
    const indices = [_]i32{ 0, 0, 0, 0, 0, 0 };
    sink.appendHost(.decode, 0, 3, &ids, &indices, 2, 6);
    try testing.expectEqual(@as(usize, 0), sink.records.items.len);
}

test "router_trace: invalid expert id marks malformed and records nothing" {
    const a = testing.allocator;
    var sink = try RouterTraceSink.init(a, testing.io, 4, ".", 4);
    defer sink.deinit();
    const ids = [_]u32{1};
    const indices = [_]i32{ 1, 2, 3, 4, 5, 900 };
    sink.appendHost(.decode, 0, 3, &ids, &indices, 1, 6);
    try testing.expect(sink.malformed);
    try testing.expectEqual(@as(usize, 0), sink.records.items.len);
}

test "router_trace: flushOnce atomic rename + idempotent" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const rel = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(rel);
    var sink = try RouterTraceSink.init(a, testing.io, 99, rel, 4);
    defer sink.deinit();
    const ids = [_]u32{ 1, 2 };
    const indices = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    sink.appendHost(.prefill, 0, 10, &ids, &indices, 2, 6);
    sink.seal(.{ .finish_reason = "stop", .prompt_tokens = 2, .completion_tokens = 4 });
    try sink.flushOnce();
    try sink.flushOnce();
    const st = try tmp.dir.statFile(testing.io, "99.bin", .{});
    try testing.expectEqual(@as(u64, 48), st.size);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(testing.io, "99.bin.tmp", .{}));
}

test "router_trace: k == 0 marks malformed, no panic, no records" {
    const a = testing.allocator;
    var sink = try RouterTraceSink.init(a, testing.io, 5, ".", 4);
    defer sink.deinit();
    const ids = [_]u32{1};
    const indices = [_]i32{ 1, 2, 3, 4, 5, 6 };
    sink.appendHost(.prefill, 0, 10, &ids, &indices, 1, 0);
    try testing.expect(sink.malformed);
    try testing.expectEqual(@as(usize, 0), sink.records.items.len);
}

test "router_trace: k == 7 marks malformed, no records" {
    const a = testing.allocator;
    var sink = try RouterTraceSink.init(a, testing.io, 6, ".", 4);
    defer sink.deinit();
    const ids = [_]u32{1};
    const indices = [_]i32{ 1, 2, 3, 4, 5, 6, 7 };
    sink.appendHost(.prefill, 0, 10, &ids, &indices, 1, 7);
    try testing.expect(sink.malformed);
    try testing.expectEqual(@as(usize, 0), sink.records.items.len);
}

test "router_trace: layer_id > 255 marks malformed, no records" {
    const a = testing.allocator;
    var sink = try RouterTraceSink.init(a, testing.io, 7, ".", 4);
    defer sink.deinit();
    const ids = [_]u32{1};
    const indices = [_]i32{ 1, 2, 3, 4, 5, 6 };
    sink.appendHost(.prefill, 0, 256, &ids, &indices, 1, 6);
    try testing.expect(sink.malformed);
    try testing.expectEqual(@as(usize, 0), sink.records.items.len);
}

test "router_trace: flushOnce true no-rewrite idempotence" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const rel = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(rel);
    var sink = try RouterTraceSink.init(a, testing.io, 97, rel, 4);
    defer sink.deinit();
    const ids = [_]u32{ 1, 2 };
    const indices = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    sink.appendHost(.prefill, 0, 10, &ids, &indices, 2, 6);
    sink.seal(.{ .finish_reason = "stop", .prompt_tokens = 2, .completion_tokens = 4 });
    try sink.flushOnce();

    const before = try readFileAllocTest(a, tmp.dir, "97.bin");
    defer a.free(before);

    sink.records.items[0] = 0xAA;
    try sink.flushOnce();

    const after = try readFileAllocTest(a, tmp.dir, "97.bin");
    defer a.free(after);
    try testing.expectEqualSlices(u8, before, after);
}

fn readFileAllocTest(allocator: std.mem.Allocator, dir: std.Io.Dir, path: []const u8) ![]u8 {
    const f = try dir.openFile(testing.io, path, .{});
    defer f.close(testing.io);
    var rb: [1024]u8 = undefined;
    var rs = f.reader(testing.io, &rb);
    return rs.interface.allocRemaining(allocator, .unlimited);
}
