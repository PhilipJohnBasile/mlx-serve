const std = @import("std");

const checkpoint_index_limit: usize = 64 * 1024 * 1024;

// These are independently loaded root sidecars, not obsolete trunk shards.
// Keep them in the indexed selection so loader behavior and byte accounting
// stay aligned when a checkpoint index does not mention them.
const supplemental_root_files = [_][]const u8{
    "mtp.safetensors",
    "model-mtp.safetensors",
};

pub const IndexedRootFiles = struct {
    names: [][]u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *IndexedRootFiles) void {
        for (self.names) |name| self.allocator.free(name);
        self.allocator.free(self.names);
    }
};

fn isRootSafetensorsBasename(name: []const u8) bool {
    return name.len > ".safetensors".len and
        std.mem.endsWith(u8, name, ".safetensors") and
        !std.fs.path.isAbsolute(name) and
        std.mem.indexOfScalar(u8, name, '/') == null and
        std.mem.indexOfScalar(u8, name, '\\') == null;
}

fn appendUnique(
    allocator: std.mem.Allocator,
    seen: *std.StringHashMapUnmanaged(void),
    names: *std.ArrayList([]u8),
    name: []const u8,
) bool {
    if (seen.contains(name)) return true;
    seen.put(allocator, name, {}) catch return false;
    const owned = allocator.dupe(u8, name) catch return false;
    names.append(allocator, owned) catch {
        allocator.free(owned);
        return false;
    };
    return true;
}

/// Resolve the unique root files selected by a valid checkpoint index, plus
/// recognized root sidecars loaded independently by the runtime. Null means
/// the index is absent or cannot be trusted, so callers must use the legacy
/// all-root-safetensors behavior instead.
pub fn indexedRootFiles(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
) ?IndexedRootFiles {
    var file = dir.openFile(io, "model.safetensors.index.json", .{}) catch return null;
    defer file.close(io);
    var read_buf: [8192]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    const bytes = reader.interface.allocRemaining(allocator, .limited(checkpoint_index_limit)) catch return null;
    defer allocator.free(bytes);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const weight_map = parsed.value.object.get("weight_map") orelse return null;
    if (weight_map != .object or weight_map.object.count() == 0) return null;

    var names: std.ArrayList([]u8) = .empty;
    var keep_names = false;
    defer if (!keep_names) {
        for (names.items) |name| allocator.free(name);
        names.deinit(allocator);
    };
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(allocator);

    var it = weight_map.object.iterator();
    while (it.next()) |entry| {
        const value = entry.value_ptr.*;
        if (value != .string or !isRootSafetensorsBasename(value.string)) return null;
        if (seen.contains(value.string)) continue;
        const st = dir.statFile(io, value.string, .{}) catch return null;
        if (st.kind != .file) return null;
        if (!appendUnique(allocator, &seen, &names, value.string)) return null;
    }

    for (&supplemental_root_files) |name| {
        const st = dir.statFile(io, name, .{}) catch continue;
        if (st.kind != .file) continue;
        if (!appendUnique(allocator, &seen, &names, name)) return null;
    }

    const owned_names = names.toOwnedSlice(allocator) catch return null;
    keep_names = true;
    return .{ .names = owned_names, .allocator = allocator };
}

fn addWeightBytes(total: *u64, size: u64) bool {
    const sum = @addWithOverflow(total.*, size);
    if (sum[1] != 0) return false;
    total.* = sum[0];
    return true;
}

fn allRootSafetensorsBytes(io: std.Io, dir: std.Io.Dir) ?u64 {
    var it = dir.iterate();
    var total: u64 = 0;
    var found_any = false;
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind != .file) continue;
        if (!addWeightBytes(&total, @intCast(st.size))) return null;
        found_any = true;
    }
    return if (found_any) total else null;
}

/// Weight bytes for the files the runtime will load. Missing or invalid
/// indexes fall back to every root safetensors file. Symlink targets count by
/// their real size, and any invalid indexed reference rejects the whole index.
pub fn checkpointWeightBytes(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
) ?u64 {
    if (indexedRootFiles(io, allocator, dir)) |selection_value| {
        var selection = selection_value;
        defer selection.deinit();
        var total: u64 = 0;
        for (selection.names) |name| {
            const st = dir.statFile(io, name, .{}) catch return allRootSafetensorsBytes(io, dir);
            if (st.kind != .file or !addWeightBytes(&total, @intCast(st.size)))
                return allRootSafetensorsBytes(io, dir);
        }
        return total;
    }
    return allRootSafetensorsBytes(io, dir);
}

test "indexed root selection preserves recognized sidecars and excludes unrelated weights" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "0123456789" });
    try tmp.dir.writeFile(io, .{ .sub_path = "unused.safetensors", .data = "unused" });
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp.safetensors", .data = "mtp" });
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"weight_map\":{\"a\":\"model.safetensors\",\"b\":\"model.safetensors\"}}",
    });

    var selection = indexedRootFiles(io, allocator, tmp.dir).?;
    defer selection.deinit();
    try std.testing.expectEqual(@as(usize, 2), selection.names.len);
    try std.testing.expectEqualStrings("model.safetensors", selection.names[0]);
    try std.testing.expectEqualStrings("mtp.safetensors", selection.names[1]);
    try std.testing.expectEqual(@as(?u64, 13), checkpointWeightBytes(io, allocator, tmp.dir));
}
