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

# DeepSeek V4 Flash 0731 — MLX M5 Max Target-Only (Experimental)

This is an **experimental, target-only MLX derivative** of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
pinned to upstream revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

It is intended for public testing on a 128 GB Apple Silicon Mac. It has passed
real M5 Max 128 GB load-and-generation smokes, but it has **not** yet passed a
fresh blind quality evaluation or a representative performance benchmark.
Do not interpret this upload as a no-quality-loss or speed claim.

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

We ran the same four public, deterministic prompts against this artifact and
the same DeepSeek-V4-Flash-0731 model served by OpenRouter's pinned CoreWeave
FP8 endpoint. Provider fallbacks were disabled. This is a small behavioral
regression gate, not a reproduction of DeepSeek's agent benchmarks and not a
full-logit or source-exactness claim.

| Public case | OpenRouter original | This MLX artifact | Comparison |
| --- | ---: | ---: | --- |
| exact single-token instruction | pass | pass | byte-identical output |
| punctuation/case copy | pass | pass | byte-identical output |
| constrained JSON | pass | pass | parsed JSON objects equal |
| one-sentence Spanish explanation | pass | pass | both valid; first 20 generated tokens matched |

The remote four-case run cost `$0.00003432`. Timing was not sealed into this
comparison receipt, so the comparison supports no throughput claim. The
original first-party DeepSeek endpoint was unavailable under the test key's
OpenRouter data-policy settings; the remote reference was the exact model slug
on a pinned CoreWeave FP8 endpoint with no fallback.

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
branch (backend fix for measured streaming: `5df7f0c`; backend initial release:
[`0bbe062e3a25a5de8cb31f3e8948c76516ff8404`](https://github.com/PhilipJohnBasile/MTPLX/commit/0bbe062e3a25a5de8cb31f3e8948c76516ff8404)).
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
  `937d2ea844e3e84d81c74daefe610ddbea0976ec7ca0de360a6e57ffb6e28201`

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
