---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - apple-silicon
  - deepseek-v4
  - mixture-of-experts
  - experimental
---

# DeepSeek V4 Flash 0731 — MLX M5 Max Target-Only (Superseded Reference)

> **Status (2026-08-12): this MLX artifact is superseded on the M5 Max by
> the DwarfStar (antirez/ds4) GGUF build of the same model.** On this
> hardware we measured the DwarfStar build at roughly 2–3× faster decode
> with strictly better fidelity (0.39 avg token-NLL vs official 0731
> continuations, 86% top-1, 78% top-N recall, 98.7% ranking agreement,
> via DwarfStar's own `score_official`). This MLX artifact is retained as
> an **experimental, honest reference** for the MLX/`mlx-serve` path — its
> measured limits are stated below, not hidden — and as provenance
> evidence for the conversion work.

This is an **experimental, target-only MLX derivative** of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
pinned to upstream revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

It is intended for public testing on a 128 GB Apple Silicon Mac. A real
M5 Max 128 GB load-and-generation smoke passed, and a deterministic 50-task
comparison against the official model was run (results below). The 50-task
result shows the artifact matches the original on copy/JSON/code/multilingual
but regresses on arithmetic and multi-step reasoning, and it is slower than
the DwarfStar GGUF build on the same hardware. **Do not interpret this
upload as a no-quality-loss or no-speed-loss claim; prefer the DwarfStar
build on this machine.**

## What is included

- 43 target-model layers in 44 safetensor shards
- 2,320 target tensors
- 103,848,946,780 tensor-payload bytes
- 103,855,768,335 total logical bytes in the validated local view
- serial target model only: `num_nextn_predict_layers=0`
- no MTP/DSpark drafter weights

Quantization recipe:

- expert `w1`/`w3`, layers 0–38: Q2 group 128
- expert `w2`, layers 0–38: Q3 group 128
- expert `w1`/`w2`/`w3`, layers 39–42: Q4 group 64
- attention, shared-expert, embedding, and head projections: affine Q8 group 64

## Verified hardware smoke

Observed on an Apple M5 Max with 128 GB unified memory using the companion
ReleaseFast `mlx-serve` DeepSeek-V4 runtime:

- all 2,320 tensors from all 44 shards loaded
- model reached `Model ready`
- deterministic prompt output: exactly `READY`
- first-touch prompt: 10 tokens at 0.937 tokens/s
- decode: 2 tokens at 20.326 tokens/s
- peak memory: 100.181 GB

This two-token decode is a smoke result, not a throughput benchmark. Only a
128 GB M5 Max has been tested; smaller-memory Macs are not claimed supported.

## Original-model comparison

### 50-task quality benchmark (primary)

A deterministic 50-task suite (8 exact-copy, 8 JSON, 12 arithmetic, 10
multi-step reasoning, 6 code, 3 multilingual, 3 classification) is run with
objective programmatic graders on all three arms, 3 draws per task, pass =
majority of draws, temperature 0, non-streaming, logprobs enabled. The
original is served via OpenRouter's pinned CoreWeave FP8 endpoint
(`coreweave/fp8`, no fallbacks), matching the published four-case reference.

| Arm | Graded | Pass | Rate | mean tok/s |
| --- | ---: | ---: | ---: | ---: |
| **Original DeepSeek-V4-Flash-0731 (CoreWeave FP8)** | 50 | **50** | **100%** | n/a (network-inclusive) |
| **Target-only, direct mlx-serve** | 50 | 37 | 74% | 10.9 |
| **Target-only, MTPLX-routed** | 50 | 37 | 74% | 10.2 |

By category (direct mlx-serve; MTPLX identical):

| Category | Original | Target-only | Verdict |
| --- | ---: | ---: | --- |
| exact copy | 8/8 | 8/8 | parity |
| JSON | 8/8 | 8/8 | parity |
| **math arithmetic** | **12/12** | **7/12** | **regression** |
| **multi-step reasoning** | **10/10** | **3/10** | **regression** |
| code | 6/6 | 6/6 | parity |
| multilingual | 3/3 | 3/3 | parity |
| classification | 3/3 | 2/3 | regression |

**This is a real, measured quality gap.** The 8-bit affine target-only
artifact matches the original on copy/JSON/code/multilingual but regresses on
arithmetic and multi-step reasoning (observed: 17×23 → 289 not 391; Tom's 3-1
apples doubled → 5 not 4; vowels in "hello" → 4 not 2). There is **no speed
penalty**: local throughput is genuine MLX engine tok/s (~10-11), while the
remote figure is network-inclusive and is not model speed. The direct and
MTPLX paths are byte-identical (same delegated engine).

Receipt: `receipts/dsv4/2026-08-12-50task-three-arm-comparison.json`;
per-run receipts `receipts/dsv4/quality-speed-20260812T030200Z.json`
(remote 50/50) and `quality-speed-20260812T034809Z.json` (local 37/50).

### Four-case behavioral gate (secondary, superseded by the above)

The same four public deterministic prompts against the same pinned CoreWeave
FP8 endpoint produced byte-identical outputs on the two exact-copy cases,
whitespace-equal JSON, and semantically-equivalent Spanish; receipt
`receipts/dsv4/2026-08-11-three-arm-20260812T024136Z.json`. The original
four-case remote run cost `$0.00003432`.

Broader public-task evaluation remains pending. In particular, this artifact
does not claim the upstream Terminal Bench, NL2Repo, Cybergym, DeepSWE,
Toolathlon, Agents' Last Exam, AutomationBench, or DSBench scores.

## Running it

The validated baseline uses the companion native `mlx-serve` DeepSeek-V4
runtime with prompt lookup, lossy decode-attention quantization, and vision all
disabled:

```bash
mlx-serve \
  --model /path/to/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly \
  --prompt 'Reply with exactly: READY' \
  --max-tokens 4 --temp 0 \
  --no-pld --no-decode-attn-quant --no-vision \
  --ctx-size 512 --timeout 300
```

Experimental MTPLX support is available via the `codex/deepseek-v4-mlxserve-backend`
branch (backend release + gate/streaming fixes: `14413c2`; backend initial
release: [`0bbe062e3a25a5de8cb31f3e8948c76516ff8404`](https://github.com/PhilipJohnBasile/MTPLX/commit/0bbe062e3a25a5de8cb31f3e8948c76516ff8404)).
It delegates to the companion native `mlx-serve` DeepSeek-V4 runtime and keeps
this target-only artifact on the AR path. The backend sets `MLX_SERVE_WIRED=fit`
for the child by default (override with `MTPLX_DSV4_WIRED`):

```bash
mtplx pull philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly

MTPLX_MLX_SERVE_BIN=/path/to/mlx-serve \
mtplx serve \
  --model philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly \
  --no-mtp --host 127.0.0.1 --port 8000 --yes
```

Two short non-streaming MTPLX smokes completed at about 23.5 output tokens/s.
An earlier streaming collapse (0.462 tokens/s then a zero-headroom refusal) was
traced to the launch memory policy, not the model: with `MLX_SERVE_WIRED=off`
the unwired 100 GB working set thrashes. Under the `fit` wired-residency policy
(`MLX_SERVE_WIRED=fit`, `MLX_SERVE_WIRED_SLACK_MB=0`), the same server streams
over HTTP at 28.3 decode tokens/s with no swap activity and 100.185 GB peak
memory. Use that policy when serving:

```bash
MLX_SERVE_WIRED=fit MLX_SERVE_WIRED_SLACK_MB=0 \
MTPLX_MLX_SERVE_BIN=/path/to/mlx-serve \
mtplx serve \
  --model philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly \
  --no-mtp --host 127.0.0.1 --port 8000 --yes
```

Representative multi-request stress and long-context streaming still warrant a
dedicated load test before any throughput claim is published.

## Provenance

- local materialization SEAL SHA-256:
  `50cd20ae84b6c7ebe79c27e08da89c3e419fcde5b529fbd6c47f6387bbf0e79f`
- target-only input contract SHA-256:
  `d69c6fa36d909d0bfc964324fac00e793798088bfffc71528d96c10cb45a4b3c`
- reviewed materializer SHA-256:
  `55d053b4daee2556aef30ea41993acde660c759a070e2af5156c7dd5af40d275`
- materializer tests SHA-256:
  `83f0c3916e95c02bfd623fd9c05343d2b87e9a9d781709730dddd31b75adcb53`
- tested ReleaseFast runtime binary SHA-256:
  `5c85c7cde0c4e1c1a9960e2ea1d9b48ada2c4f20f23b3c1617c9e9a5bb633d42`
  (includes the JSON logprobs UTF-8 fix: `jsonEscape` now escapes bytes that
  would otherwise form invalid UTF-8 inside a JSON string, so byte-level-BPE
  tokens that decode to a partial multi-byte sequence — e.g. the first two
  bytes of an ellipsis — no longer corrupt the response. This removes the
  intermittent `UnicodeDecodeError` that previously dropped a task from the
  benchmark receipt.)

The source model is licensed under MIT; see `LICENSE` and the upstream model
card for attribution and its original terms.

## Test feedback

When reporting a result, please include:

- Mac model and unified-memory capacity
- macOS version
- runtime and model revision
- exact flags and context length
- whether the failure happened during load, prefill, or decode
- peak memory and exact generated output
