# DSV4 PCTree — parent-conditioned draft tree

Status: CPU-only research prototype. No tree-shaped target verifier or serving
path is implemented. The miniature-fixture promotion gate remains **NO-GO**.

This is an independent implementation of Algorithm 1 in the primary PCTree
paper, not a copy of an author implementation:
[PCTree, arXiv:2608.02123](https://arxiv.org/html/2608.02123), §4.1–4.2.
The paper is the authority where this document and an earlier local receipt
differ.

## Contract

For block logits `L[d]`, a frontier parent `p`, and the checkpoint's Markov
head, the planner computes:

```text
z_d(p)       = L[d] + Markov(p)
logpi_d(v|p) = log_softmax(z_d(p))[v]
s(child)     = s(parent) + logpi_d(child|p)
```

At every stage, it keeps the **local top-k children of every frontier parent
in the global candidate pool**. It then selects only the current layer's best
`k` nodes as the next expansion frontier. After all stages, it keeps the
global top `N` pool nodes by `(path score descending, depth ascending, stable
ID ascending)`. This gives a prefix-closed output tree and the paper's node
bound:

```text
1 + k + (B - 1) * k^2
```

The prior `1 + B*k` beam-frontier implementation was not PCTree. Its receipt
is preserved unchanged as adverse historical evidence; see the correction
receipt below.

## Exact k=1 regression contract

The DSpark host loop chooses the first index that is strictly greater than the
current best **raw f32 logit**. PCTree therefore selects `k=1` before
log-softmax. Ranking normalized f32 log-probabilities can collapse two logits
that differ by one ULP and change the emitted token.

The planner follows the runtime even for unusual values:

- if row zero is NaN, index zero wins because every strict comparison is false;
- a later `+inf` wins, and equal `+inf` values keep the first index;
- an all-`-inf` row selects index zero and records `-inf`, never NaN.

Branching rows (`k > 1`) reject NaN or infinity with
`NonFiniteBranchLogit`. DSpark specifies only greedy top-1 behavior for such
rows, so inventing a tree order would not be source-faithful.

## Artifacts

| File | Purpose |
|---|---|
| `src/pctree.zig` | CPU candidate-pool planner, retention, mask, retrieve paths |
| `src/pctree_batch.zig` | Dense CPU reference for row-wise Markov top-k |
| `src/deepseek_v4.zig` | Default-off capture hook and serial branch oracle tests |
| `receipts/dsv4-pctree/` | Historical preregistration/result plus append-only correction |

`Plan.retrieve_rows` enumerates every retained root-to-leaf path. The
miniature oracle evaluates every such path; it does not discard a retained
short leaf merely because it is shorter than the draft block.

## Safety and lifecycle

The serial branch helper fails before allocation or `dsparkBeginWith` when the
serial verifier is not armed. It checks zero vocabulary, branch geometry,
token range, and checked shape arithmetic. The DSV4_MINI integration test
proves disabled-verifier rejection leaves a prefilled state fingerprint
unchanged. It also constructs a target-continuation proposal to exercise both
positive partial acceptance and full acceptance through the existing serial
verify/rollback/finish path, checking the resulting state against independent
serial replay.

Fixture-backed PCTree tests skip only when `DSV4_MINI` is absent. If the
variable is explicitly set, unreadable fixture files and missing required
PCTree fields are hard test failures. A negative probe must use an isolated
Zig cache, or otherwise prove the test executable reran: changing
`DSV4_MINI` while reusing the normal cache produced a false exit-0 result.
That cached result is **NOT EVIDENCE**. The isolated-cache probe exited 1 with
23/25 tests passing and both PCTree fixture-backed tests failing as required.

`k=1` token identity is proven against the runtime on the miniature fixture.
It is not a proof of real-checkpoint quality, speed, or tree-verifier
correctness.

## Current evidence and limits

The corrected planner's DSV4_MINI run kept six, seven, and six retained leaf
paths at prefixes 6, 9, and 12 respectively. None happened to be full-depth
under the small `N = 1 + B*k` budget, and every evaluated path accepted zero
draft tokens. That is still a negative miniature result, but it is no longer
valid to describe the old beam-frontier result as a PCTree-wide conclusion.

The following remain deliberately out of scope:

- an ancestor-masked, packed tree target forward;
- a real-checkpoint acceptance or speed claim;
- GPU top-k/planner kernels;
- release or default-on behavior.

## Reproduction and provenance

Use the repository's shared local toolchain and only the documented miniature
fixture:

```sh
./.zig-toolchain/zig test src/pctree.zig
./.zig-toolchain/zig test src/pctree_batch.zig
DSV4_MINI=/tmp/dsv4-mini ./.zig-toolchain/zig build test \
  -Dtest-filter=PCTree -Doptimize=ReleaseFast --summary all
DSV4_MINI=/tmp/dsv4-mini ./.zig-toolchain/zig build test \
  -Dtest-filter=DSpark -Doptimize=ReleaseSafe --summary all
```

The exact toolchain, dependencies, fixture hashes, and command results are in
`phase-4-serial-branch-acceptance-correction-20260810.json`. The worktree's
`.zig-toolchain` and `lib/mlx` entries are local symlinks and are intentionally
excluded from version control.
