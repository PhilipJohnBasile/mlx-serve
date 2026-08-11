# DeepSeek V4 Flash 0731 target-only: streaming memory-policy A/B

Status: RESOLVED — the reported streaming instability was a launch-policy
defect, not a model defect. Measured on the M5 Max 128 GB with the
target-only Gold view.

## Background

The HF card and MTPLX summary previously reported streaming instability:
a representative streaming attempt decoded at 0.462 tokens/s and then the
next request was refused with approximately zero Metal request headroom.
That made the release "experimental" with no representative throughput
claim.

Constraint math (measured):

- model logical bytes: 103,848,946,780 (44 shards, 2,320 tensors)
- peak active memory during smoke: 100.185 GB
- Metal `max_recommended_working_set_size`: 119.140625 GiB
- `iogpu.wired_limit_mb`: 122,000 (122 GB)
- transient headroom above ~100 GB active: ~19 GiB

## A/B results

Both runs: same binary (ReleaseFast `937d2ea8…`), same model dir, HTTP
SSE `/v1/chat/completions`, 4096 ctx, `max_tokens=64`, `temp=0`, pinned
cache limit 8 GiB (`MLX_SERVE_CACHE_LIMIT`).

`MLX_SERVE_WIRED=off` (previous default, hardcoded in the MTPLX backend):

- server preflight REFUSED the load: "weights ~96.7 GB, available
  102.05 GB" (page cache / compressed pressure left no headroom)
- the earlier observed failure mode under less pressure: unwired weights
  get evicted/re-established per command buffer; streaming allocations on
  top of ~100 GB active → pageout thrash → 0.46 tok/s then zero-headroom
  refusal

`MLX_SERVE_WIRED=fit` (`MLX_SERVE_WIRED_SLACK_MB=0`):

- model loaded and admitted
- streamed SSE completion: prefill 1.8 tok/s, decode 28.28 tok/s
- peak memory during decode: 100.185 GB
- no swap-in/swap-out activity during the session
- clean shutdown

Offline single-prompt streaming under `fit` also measured 31.6 tok/s
(19 generated tokens), peak 100.185 GB.

## Root cause

`WIRED=off` leaves the ~100 GB weight set unwired. Metal re-establishes
residency for the whole resident set on command-buffer boundaries and
macOS may evict/compress pages under pressure; every streamed request
allocates per-token buffers on top, crossing into pageout thrash.
`fit` wires exactly the live set (weights) with slack 0 and keeps
transients on the unwired commit-free path, so decode stays compute-bound.

## Changes

- MTPLX backend `mtplx/backends/deepseek_v4_mlxserve.py`: child default
  `MLX_SERVE_WIRED=fit`; caller override via `MTPLX_DSV4_WIRED` (consumed,
  not passed through); `MLX_SERVE_*` inheritance still filtered.
- MTPLX commit: `5df7f0c` (branch `codex/deepseek-v4-mlxserve-backend`).
- Tests after fix: 2838 passed, 11 skipped (full suite); backend file
  tests 16/16; ruff clean.
- mlx-serve `scripts/run_dsv4_0731_target_only_gold.sh`: defaults
  `MLX_SERVE_WIRED=fit`, `MLX_SERVE_WIRED_SLACK_MB=0`.
- HF model card updated (commit `bcf68a34…`): documents the fit policy
  and measured 28.3 tok/s streaming decode; README byte-verified
  identical to the local reviewed copy; 44 weight shards untouched.

## Still open (honest)

- multi-request concurrency stress and long-context streaming at 8192+
  ctx have not been re-measured after the policy fix
- representative multi-feedback load crucially still an open gate