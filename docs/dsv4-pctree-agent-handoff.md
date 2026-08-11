# DSV4 PCTree — handoff

Date: 2026-08-10. Candidate: `branch:dsv4-pctree-mlx-20260809`, frozen baseline
`68acdba6cb464268a5c2dc83188d9ce3d0a51b36`. Nothing is committed or published.

Start with [the design](dsv4-pctree-design.md) and the append-only correction
receipt. The planner is an independent implementation of [PCTree Algorithm
1](https://arxiv.org/html/2608.02123), not a copied reference implementation.

## Current state

| Artifact | Verified status |
|---|---|
| `src/pctree.zig` | 13/13 tests, including one-ULP, NaN, bounds, and allocation-failure coverage |
| `src/pctree_batch.zig` | 17/17 tests, including raw top-1 and non-finite branch guards |
| PCTree DSV4_MINI run | 25/25 total, 5/5 filtered ReleaseFast tests passed with an isolated per-run cache |
| DSpark DSV4_MINI run | 35/35 total, 15/15 filtered ReleaseSafe tests passed with an isolated per-run cache |
| PCTree target verifier | Not implemented; no serving behavior changed |

## Semantics that must stay pinned

- `k=1` uses the serial DSpark raw-f32 strict-greater selector, not normalized
  log-probabilities. It must preserve one-ULP near ties and row-zero NaN.
- `k>1` retains every local top-k child in the candidate pool, then chooses a
  width-k next frontier from that layer. The node bound is
  `1 + k + (B - 1)k^2`.
- Non-finite `k>1` rows fail closed. `k=1` preserves the runtime's unusual
  NaN/infinity behavior without creating NaN scores.
- Retention is global top-N and prefix-closed. Every retained leaf is a valid
  verifier candidate, even when shorter than the nominal block depth.
- The serial-branch oracle must reject an unarmed verifier before any state
  mutation and compare every successful state to independent serial replay.
- An absent `DSV4_MINI` skips fixture-backed tests. Once it is explicitly set,
  fixture I/O and required PCTree fields fail closed instead of skipping.

## Commands

```sh
./.zig-toolchain/zig fmt --check src/pctree.zig src/pctree_batch.zig src/deepseek_v4.zig
./.zig-toolchain/zig test src/pctree.zig
./.zig-toolchain/zig test src/pctree_batch.zig
DSV4_MINI=/tmp/dsv4-mini ./.zig-toolchain/zig build test \
  -Dtest-filter=PCTree -Doptimize=ReleaseFast --summary all
DSV4_MINI=/tmp/dsv4-mini ./.zig-toolchain/zig build test \
  -Dtest-filter=DSpark -Doptimize=ReleaseSafe --summary all
```

When testing a changed or deliberately broken `DSV4_MINI`, use a unique cache:

```sh
DSV4_MINI="$PROBE/fixture" ./.zig-toolchain/zig build test \
  -Dtest-filter=PCTree -Doptimize=ReleaseFast \
  --cache-dir "$PROBE/zig-cache" --summary all
```

A normal-cache negative probe returned exit 0 after `DSV4_MINI` changed. That
was stale-cache reuse and is **NOT EVIDENCE**. The isolated-cache rerun exited
1 with 23/25 tests passing; the two PCTree fixture-backed tests failed as
required. Changing the environment variable alone is not a sufficient
negative-test protocol.

`zig build` reports a known harmless `pkg-config --list-all` / missing
`lib/llama/lib` warning in this isolated worktree. The test command's status
and test counts—not that warning—are the result.

## Evidence discipline

The historical preregistration and original beam-frontier result are retained
unchanged. They do **not** establish a PCTree-wide no-go because the old
planner was not Algorithm 1. The corrected DSV4_MINI run also has zero
accepted draft tokens across the evaluated retained leaves. It is a negative
random-mini measurement only, not real-model acceptance, speed, or release
evidence.

Do not implement a tree-shaped verifier unless a separately preregistered,
real-geometry experiment clears its acceptance and lifecycle gates. The
complete toolchain, dependency, source, and fixture provenance is in the
correction receipt.

## Dependency provenance

The measured dependency source worktrees were not clean. MLX was detached at
`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247` with 18 tracked modifications
and three untracked kernel sources. MLX-C was detached at
`fba4470b89073180056c9ea46c443051375f7399` with two tracked modifications.
The correction receipt records the exact binary-patch, porcelain-status, and
untracked-content hashes. Those states match the same-day checkpoint.

The tested `libmlx.dylib` is pinned by SHA-256, but reproduction from the clean
base commits alone is not proven. Do not describe the dependency worktrees as
clean or the library artifact as clean-commit reproducible.

## Local-only entries

`.zig-toolchain` and `lib/mlx` are worktree-local symlinks to the main
checkout. Leave them untracked and do not rebuild, remove, stage, or publish
them from this worktree.
