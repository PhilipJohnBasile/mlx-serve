# PR #302: forced-depth Qwen3.8 comparison

This directory records a fresh response to the [benchmark review](https://github.com/ddalcu/mlx-serve/pull/302#issuecomment-5466196054).
The earlier August 29 numbers did not establish an improvement. A local September 3 rerun also did not meet the request: it substituted 4K for 2K, all six 32K cells returned HTTP 400, and the first control had a large warmup effect. Those runs are excluded from this new session, not selectively mixed into its results.

## Protocol fixed before measurement

- Control A: `6787142`, the PR's original parent; candidate B: `1032fba`, its original head. Both are rebuilt as ReleaseFast with Zig `0.17.0-dev.1818+7051f8e73` and identical staged MLX libraries. No production source is changed for this rerun.
- Model: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP`, snapshot `6a052d93a4618c8a559f402a16e8a190a48f209e`. The manifest records the actual files' SHA-256 hashes, including the MTP sidecar; the pack has mixed 4/6/8-bit trunk tensors and a BF16 MTP sidecar, according to its config and sidecar manifest. It is not a uniformly 6-bit checkpoint.
- One boot, one uninterrupted session on AC power. Full blast fans throughout measurement; preserve the observed fan checks with the results.
- Sequence: **A B B A A B B A**, four independent server launches and four measured reports per arm.
- Both arms: `MLX_SERVE_MTP_FORCE_DEPTH=8`, `MLX_SERVE_MTP_TRACE=1`, `--mtp --mtp-depth 8 --no-pld --no-vision --prefix-cache-entries 0 --log-level debug`.
- Disabling PLD isolates MTP; disabling the prefix cache prevents reused prompts from turning prefill timing into cache-hit timing. Both changes apply equally to both arms.
- `--ctx-size 40960` leaves room for the 32K rung's calibration/template overshoot and generated tokens. The measured rungs remain **2K, 8K, 16K, 32K**.
- Pinned command: `npx --yes llmprobe@0.6.7 localhost:11234 --bench-only --rungs 2k,8k,16k,32k --no-save --no-color --save <report.json>`. The earlier 0.6.1 CLI rejected 2K. Use the same 0.6.7 bundle throughout.
- Before each report, perform the same two discarded warmup requests (short and long archive prompts, 192 generated tokens each), saving the raw responses separately. Retain llmprobe's standard scenario warmups and sample counts. Each context rung has one measured sample per report, four per arm overall.
- Acceptance checks require all four rungs to succeed and be within 15% of their token targets, positive `m_avg=8.00` MTP traces in every server log, NAX enabled, and `[sdpa-split] ... qL=9` present in every A log and absent from every B log. Record drift warnings and adverse results without dropping runs.

## Reproduction

Use `tests/bench_pr302_abba.py --help` for the binary, model, new output directory, and observed fan-note arguments. Port 11234 must already be free; the harness stops only the server process it creates. The harness records commands, model and binary hashes, boot identity, AC power, and per-run engagement evidence in `manifest.json`.

The JSON files are llmprobe's raw reports. Its short-prompt `bench.speculative.novelTokPerSec` and context-rung `decodeTokPerSec` are the relevant novel-generation measurements with forced MTP active. Predictable rates and tokens per step are retained to expose acceptance variability. Prefill and TTFT movement must be treated as a control for noise, not a benefit attributable to a verify-only routing change.

This is a forced-depth diagnostic, not a measurement of the shipping adaptive controller. Width-8 split entries during startup are expected in both arms; the treatment concerns only width 9. llmprobe may classify its short-prompt SSE write counts as buffered, so those counts are not used as proof of MTP depth; the native `m_avg=8.00` traces provide that evidence.

## Results and interpretation

**These measurements do not establish a performance benefit for the PR.** Short-prompt novel speculative throughput is 20.0 tok/s for control versus 19.6 tok/s for the PR (medians across four reports). Context-rung median changes are **−5.0% at 2K, −7.0% at 8K, −4.9% at 16K, and −14.7% at 32K**. The complete per-run values and ranges are in [the generated summary](session-01/summary.md). All eight reports are retained, including the PR's faster 40.2 tok/s 32K sample.

The ranges overlap at every rung. Short-prompt prefill moved from a 735.9 tok/s control median to 631.8 tok/s on the PR, even though this routing change does not directly change prefill. Run 05-A was flagged **degraded** (−12.4%); the other seven runs were classified steady, with drift between −9.8% and +8.7%. **Concurrent host work was observed**, including Flutter tests and frontend/native builds overlapping the session; see [host-load observations](host-load-observations.json). The host was not exclusive to this experiment. These limitations prevent attributing the measured slowdowns solely to the PR.

The routing question is resolved for this session: all four controls logged width-9 split engagement, all four PR runs logged zero width-9 splits, and every run logged positive depth-8 MTP traces. All **32/32 context rungs** succeeded. The same boot and AC power were verified before and after every run, and the Full blast preset remained selected throughout the session ([fan observations](fan-checks.json)). The original Automatic preset was restored after the final benchmark completed.

This supersedes the earlier unsupported speedup interpretation. A causal performance claim would require a quiet-host comparison; these results do not justify promoting the PR as a speed improvement. PR #302 was already closed by the maintainer on September 12; this evidence does not reopen it.

## Engagement excerpts

Control A (01-A; each control has its own equivalent line):

```text
[sdpa-split] engaged: qL=9 kL=2369 Hq=24 Hkv=4 (MLX_SERVE_SDPA_SPLIT=0 restores the single dispatch)
```

PR B (02-B; no width-9 split line in any PR log):

```text
[nax-sdpa] engaged: stock fused sdpa (force_fused) serves hd-256 causal prefill and verify blocks > 8 rows; msv_attn_p256 causal arm declined (MLX_SERVE_NAX_SDPA=0 restores)
  [mtp-trace] rounds=32 avg_ms draft=2.93 sync=0.00 ext=0.00 verify=2.49 corr=0.00 eval=89.54 hist=0.15 commit=0.08 predraft=40.26 gap=0.05 total=135.50 | m_avg=8.00 acc_avg=2.44 ext_rate=0.00 acc_idx=0.72/0.50/0.38/0.28/0.22/0.13/0.13/0.09
```

[Per-run engagement counts](session-01/summary.md#engagement) and full logs accompany the reports. The MTP depth trace is positive evidence; absence of a split line alone is not treated as proof.

## Artifacts and validation

- [Download the complete evidence ZIP](evidence.zip): all eight raw llmprobe JSONs, full server/console/probe logs, discarded warmup responses, manifests, summaries, hashes, and host/fan observations.
- [Manifest with exact commands and hashes](session-01/manifest.json), [final integrity check](session-01/final-integrity.json), [llmprobe bundle identity](llmprobe-tool.json), and [build/test receipt](build/validation.json).
- Both original revisions rebuilt successfully in ReleaseFast with the pinned compiler. Focused SDPA parity/routing tests exited 0; all five harness regression checks passed. The guard rejects the old missing-2K/failed-32K evidence. This is not a full application release or a whole-suite correctness qualification.
- All checkpoint, binary, and MLX-library hashes matched after the final run. No production source changed during this rerun. Zero cached prompt tokens were reported by all eight prefix-cache probes.

| Order | Arm | Raw llmprobe JSON |
|---|---|---|
| 1 | A | [01-A.llmprobe.json](session-01/01-A.llmprobe.json) |
| 2 | B | [02-B.llmprobe.json](session-01/02-B.llmprobe.json) |
| 3 | B | [03-B.llmprobe.json](session-01/03-B.llmprobe.json) |
| 4 | A | [04-A.llmprobe.json](session-01/04-A.llmprobe.json) |
| 5 | A | [05-A.llmprobe.json](session-01/05-A.llmprobe.json) |
| 6 | B | [06-B.llmprobe.json](session-01/06-B.llmprobe.json) |
| 7 | B | [07-B.llmprobe.json](session-01/07-B.llmprobe.json) |
| 8 | A | [08-A.llmprobe.json](session-01/08-A.llmprobe.json) |

To recompute the summary, extract `evidence.zip` into this directory and run `python3 tests/summarize_pr302_abba.py benchmarks/pr302/2026-09-13/session-01` from the repository root.

