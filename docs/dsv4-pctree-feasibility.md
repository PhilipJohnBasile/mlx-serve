# DeepSeek V4 PCTree feasibility

Date: 2026-08-10. This report concerns the isolated CPU research worktree only.
It does not authorize a serving change.

## Can the candidate tree be planned correctly?

**Yes, for the tested CPU contract.** The planner now follows the primary
[PCTree Algorithm 1](https://arxiv.org/html/2608.02123): all local top-k
children enter the candidate pool; only the next expansion frontier is capped
at k; then global top-N retention creates the packed prefix-closed tree. This
is an independent implementation with 13/13 planner tests and 17/17 batched
reference tests.

The previous beam-frontier variant was narrower than PCTree. Its historical
zero-delta receipt remains intact, but the correction receipt prevents it
being misrepresented as a general PCTree result.

## Does k=1 reproduce serial DSpark?

**Yes, for the tested miniature CPU-oracle path.** The selector now uses raw
f32 logits in DSpark's strict-greater order rather than normalized log
probabilities. Tests cover a one-ULP near tie, row-zero NaN, later NaN, repeated
infinity, and an all-negative-infinity row. The DSV4_MINI oracle matches the
runtime draft IDs at prefixes 6, 9, and 12; the PCTree-filtered ReleaseFast run
passed 5/5, including a negative required-fixture-field contract test. An
explicitly set `DSV4_MINI` now propagates fixture I/O and required-field
failures rather than treating them as skips. This negative gate must use an
isolated Zig cache (or otherwise force execution): ordinary cache reuse once
returned exit 0 after the fixture path changed and is retained as **NOT
EVIDENCE**. The isolated-cache rerun exited 1, with 23/25 passing and both
PCTree fixture-backed tests failing as required.

This is token-id and state-contract evidence for the test fixture. It is not
a claim that a future packed target forward will be byte-identical.

## Does k>1 improve acceptance?

**No positive evidence on DSV4_MINI.** With the corrected candidate-pool
planner and the fixed small retention budget, the retained leaves at prefixes
6, 9, and 12 were all shorter than the nominal block and all accepted zero
draft tokens. Every retained leaf was evaluated through the serial B1–B5
verifier and independent-state-replay gate. The best accepted-length delta
remained zero at all three prefixes.

This does not prove PCTree cannot help a trained, real checkpoint. It proves
only that the miniature fixture has not earned the complexity of a tree-shaped
verifier.

## Is the dependency provenance clean-commit reproducible?

**No.** The tested MLX and MLX-C source worktrees contained disclosed tracked
patches, and MLX also contained three untracked kernel sources. The correction
receipt pins their base commits, deterministic binary-patch hashes, porcelain
status hashes, untracked source-content hashes, and the tested
`libmlx.dylib`. The artifact hash identifies what was tested; it does not prove
that the clean base commits alone reproduce that library.

## Is tree verification feasible now?

**No-go for implementation.** No ancestor-masked target verifier, packed
tree cache policy, or full-model quality/speed receipt exists. The prototype
may be revisited only after a separately preregistered real-model experiment
shows a reproducible positive acceptance benefit and retains exact lifecycle
evidence.

| Gate | Status |
|---|---|
| Algorithm-1 CPU candidate pool | Pass on unit/reference tests |
| k=1 serial selector | Pass on DSV4_MINI CPU oracle |
| Positive k>1 acceptance | Not established; corrected mini result is zero delta |
| Packed tree verification | Not implemented |
| Production readiness | No-go |

The historical preregistration/result and the corrected run are linked in
`receipts/dsv4-pctree/`. Read the correction receipt before drawing any
comparison from the original beam-frontier result.
