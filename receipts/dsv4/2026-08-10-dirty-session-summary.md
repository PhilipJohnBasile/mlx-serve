## 2026-08-10 Illegal-Arguments Patch Recap

### What was done

1. Reconciled the pasted ChatGPT/Codex handoff with the actual worktree.
2. Checkpointed the current dirty state into a hash-only manifest and committed it.
3. Ran the current-source build and full suite from a fresh cache. Result:
   - ReleaseFast: passed.
   - Full suite: `1333 passed`, `111 skipped`.
   - `DSV4_MINI` focused: passed, including raw-KV COW, replay-commit, B1–B5, and current runtime paths.
4. Ran the six-lane and B1–B5 serial-equivalence tests with only the intended env set. Result: passed with zero stage deltas, equal state fingerprints, and equal content hashes.
5. Ran the lifecycle harness against rebuilt Gold. Result: **fail closed** on swap/pageout:
   - `343` pageouts
   - `600,204` swapouts
   - `1,216,935,362` swap-used bytes
   - only the serial boot ran before the gate halted; no DSpark boots ran.
6. Investigated the memory topology:
   - Active model bytes remain about `107.6 GB`.
   - Available memory at ~`3.7 GB`, so a normal DSpark request is rejected before decoding.
   - The lifecycle failure is therefore a physical headroom blocker, not a remaining state-ownership leak.
7. Reconciled the authenticated benchmark env contract after the replay-commit policy became default-on. Updated the B2/E2 environment deltas to explicitly disable replay-commit, then fixed the offline self-tests that were serializing the old env shape.

### Repository state after this patch

- Commits produced:
  - `70bcb02` checkpoint current DS4 worktree
  - `841f4d6` test(dspark): pin replay-commit disables in B2/E2 env deltas
- Untouched pending work remains in the dirty worktree:
  - `src/deepseek_v4.zig`, `scheduler.zig`, `model.zig`, `server.zig`, `metrics.zig`, `mlx.zig`
  - dirty `lib/mlx-src` and `lib/mlxc-src`
  - many untracked receipts under `receipts/dsv4/`
- All old receipts and handoff files were left intact.

The progress, build/test outputs, and receipts are preserved. No model weights or release artifacts were modified or published.