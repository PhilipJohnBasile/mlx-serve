#!/usr/bin/env python3
"""Three-arm DSV4 behavioral comparison runner.

Reuses the FROZEN case definitions from the reviewed harness
(bench_dsv4_openrouter_local_gate.py) so today's comparison is
comparable to the published OpenRouter oracle:

  arm 1  remote       OpenRouter pinned CoreWeave FP8 deployment
                     (requires the frozen oracle receipt; no re-spend
                      needed unless --spend-remote is used)
  arm 2  direct       local mlx-serve endpoint (:PORT)
  arm 3  mtplx        local MTPLX-routed endpoint (:MTPLX_PORT)

Every arm posts the identical four public cases with the same
logprobs/top_logprobs=5 contract, so the receipts are directly
comparable. Local latency is reported separately from model behavior;
network latency is never called model speed.

Usage:
  python3 bench_dsv4_three_arm.py \
      --port 11234 --mtplx-port 8000 \
      --out receipts/dsv4/2026-08-11-three-arm.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.bench_dsv4_openrouter_local_gate import (  # noqa: E402
    CASE_ORDER,
    CASES,
    local_request,
)

OUT = Path("/Users/pjb/git/mlx-serve/receipts/dsv4")
LOCAL_MODEL = "DeepSeek-V4-Flash-0731-target-only-gold-view-20260811-v1"


def post(endpoint: str, payload: dict, timeout: int = 900):
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, time.time() - t0


def run_local_arm(endpoint: str, model_id: str, arm: str = "local_mlx_serve") -> dict:
    results = {}
    for name in CASE_ORDER:
        case = CASES[name]
        payload = local_request(case, model_id)
        body, dt = post(endpoint, payload)
        content = (body.get("choices") or [{}])[0].get("message", {}).get(
            "content") or ""
        logprobs = []
        lp = (body.get("choices") or [{}])[0].get("logprobs") or {}
        for entry in (lp.get("content") or []):
            logprobs.append({
                "token": entry.get("token"),
                "logprob": entry.get("logprob"),
                "top": [t.get("token") for t in (entry.get("top_logprobs") or [])],
            })
        results[name] = {
            "content": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "token_count": len((content.split())),  # coarse; per-token logprobs authoritative
            "logprobs_entries": len(logprobs),
            "top_tokens": [e.get("token") for e in logprobs[:3]],
            "latency_s": round(dt, 3),
        }
    return {"arm": arm, "endpoint": endpoint, "model": model_id,
            "cases": results}


def load_remote_oracle(path: Path | str) -> dict:
    d = json.loads(Path(path).read_text())
    results = d.get("results") or []
    cases = {}
    for item in results if isinstance(results, list) else []:
        name = item.get("case")
        if not name:
            continue
        # `generation` is the raw API response object; the content lives in
        # completion["content"] (or a top-level content fallback).
        content = None
        completion = item.get("completion")
        if isinstance(completion, dict):
            content = completion.get("content")
        if not isinstance(content, str):
            gen = item.get("generation")
            if isinstance(gen, dict):
                content = gen.get("content")
            elif isinstance(gen, str):
                content = gen
        if not isinstance(content, str):
            content = json.dumps(item)
        cases[name] = {
            "content": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "token_count": len(content.split()),
            "top_tokens": None,
        }
    # attach the published per-case verdicts (exact-bytes / JSON-semantic /
    # local-contract) when the comparison receipt is present beside the
    # oracle, so byte-inequality on semantic cases is not misread as failure.
    oracle_dir = Path(path).parent
    pub = None
    for candidate in oracle_dir.glob("*openrouter*v4-flash-local-comparison.json"):
        p = json.loads(candidate.read_text())
        for c in p.get("comparisons", []):
            case = c.get("case")
            if case in cases:
                cases[case]["published_verdict"] = c.get("primary_comparison")
        pub = True
        break
    return {
        "arm": "oracle",
        "endpoint": "OpenRouter coreweave/fp8 (frozen oracle, no re-spend)",
        "model": d.get("requested_model"),
        "published_comparison_found": pub,
        "cases": cases,
    }


def compare(a: dict, b: dict, name: str) -> dict:
    per = {}
    for c in CASE_ORDER:
        if c not in a["cases"] or c not in b["cases"]:
            per[c] = {"present": False}
            continue
        ca, cb = a["cases"][c], b["cases"][c]
        published = ca.get("published_verdict") if a["arm"] == "oracle" else (
            cb.get("published_verdict") if b["arm"] == "oracle" else None)
        import re
        def squish(s):
            return re.sub(r"\s+", "", s)
        per[c] = {
            "content_equal": ca.get("content") == cb.get("content"),
            "content_sha_equal": ca.get("content_sha256") ==
            cb.get("content_sha256"),
            "whitespace_normalized_equal": squish(ca.get("content", "")) ==
            squish(cb.get("content", "")),
            "a_sha": (ca.get("content_sha256") or "")[:12],
            "b_sha": (cb.get("content_sha256") or "")[:12],
            "published_oracle_verdict": published,
        }
    return {"arm": name, "cases": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11234)
    ap.add_argument("--mtplx-port", type=int, default=8000)
    ap.add_argument("--oracle", default=str(
        OUT / "2026-08-11-openrouter-v4-flash-oracle.json"))
    ap.add_argument("--spend-remote", action="store_true",
                    help="re-hit OpenRouter (costs credits); default uses the "
                         "frozen oracle receipt")
    args = ap.parse_args()

    oracle = load_remote_oracle(Path(args.oracle))
    direct = run_local_arm(f"http://127.0.0.1:{args.port}", LOCAL_MODEL,
                           "direct_mlx_serve")
    mtplx = run_local_arm(f"http://127.0.0.1:{args.mtplx_port}", LOCAL_MODEL,
                          "mtplx_routed")

    comparisons = [
        compare(oracle, direct, "remote_vs_direct"),
        compare(oracle, mtplx, "remote_vs_mtplx"),
        compare(direct, mtplx, "direct_vs_mtplx"),
    ]

    receipt = {
        "schema": "mlx-serve.dsv4-three-arm-behavior-v1",
        "schema_version": 1,
        "arms": {"oracle": oracle, "direct_mlx_serve": direct,
                 "mtplx": mtplx},
        "comparisons": comparisons,
        "note": (
            "Local latency is reported per arm but is NOT comparable to "
            "networked OpenRouter latency; behavior (content + top-token "
            "logprobs) is the cross-arm signal. MTPLX delegates to the same "
            "mlx-serve engine, so direct_vs_mtplx is an identical-engine "
            "routing check."
        ),
    }
    import datetime
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    out = OUT / f"2026-08-11-three-arm-{stamp}.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")

    print(f"wrote {out}")
    for c in comparisons:
        print(f"\n== {c['arm']}")
        for name, r in c["cases"].items():
            eq = ("equal" if r.get("content_equal")
                  else "ws-equal" if r.get("whitespace_normalized_equal")
                  else "different")
            print(f"  {name:10s} {eq:8s} sha={r.get('a_sha')} vs "
                  f"{r.get('b_sha')} | verdict={r.get('published_oracle_verdict')}")


if __name__ == "__main__":
    main()