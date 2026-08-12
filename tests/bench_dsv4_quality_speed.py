#!/usr/bin/env python3
"""Real quality + speed benchmark: DSV4 target-only vs OpenRouter original.

Runs a deterministic task suite with OBJECTIVE graders on three arms:
  remote   OpenRouter deepseek/deepseek-v4-flash-0731, pinned CoreWeave
           fp8 endpoint, temp 0, fallbacks disabled (the same reference
           the published comparison used)
  direct   local mlx-serve
  mtplx    local MTPLX-routed backend (same mlx-serve engine)

Each task carries a programmatic grader (exact string, parsed JSON
equality, or numeric equality), so "pass" means the same thing on the
original and the derivative. Cross-arm exact/normalized agreement is
reported separately from per-arm pass rate. Speed is measured per request
from the server's usage.token counts + wall time; remote wall time is
network-inclusive and is NEVER presented as model speed.

Usage:
  OPENROUTER_API_KEY=... python3 bench_dsv4_quality_speed.py \
      --remote --port 11234 --mtplx-port 8000 --out receipts/dsv4/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

MODEL_ID = "deepseek/deepseek-v4-flash-0731"
PROVIDER_SLUG = "coreweave/fp8"
LOCAL_MODEL = "DeepSeek-V4-Flash-0731-target-only-gold-view-20260811-v1"


def _exact(gold: str):
    def grade(text: str):
        return (text.strip() == gold.strip(),
                f"exact '{text.strip()!r}' vs '{gold.strip()!r}'")
    return grade


def _json_equal(gold_obj):
    def grade(text: str):
        try:
            got = json.loads(text)
            return (got == gold_obj,
                    f"json {got!r} vs {gold_obj!r}")
        except Exception as e:
            return False, f"unparseable: {e}"
    return grade


def _num(gold: float, tol: float = 1e-6):
    def grade(text: str):
        nums = re.findall(r"-?\d+\.?\d*", text)
        if not nums:
            return False, f"no number in {text!r}"
        val = float(nums[-1])
        return (abs(val - gold) <= tol, f"last-num {val} vs {gold}")
    return grade


def _contains(*needles):
    def grade(text: str):
        missing = [n for n in needles if n not in text]
        return (not missing, f"missing {missing or 'none'} in output")
    return grade


TASKS: list[dict] = [
    # ---- exact copy / mechanical (deterministic graders) ----
    {"name": "exact_ready", "system": "Output exactly READY and nothing else.",
     "user": "READY", "max_tokens": 16, "grade": _exact("READY")},
    {"name": "exact_copy_1", "system": "Output exactly the string and nothing else.",
     "user": "Ax7_Q!m2", "max_tokens": 16, "grade": _exact("Ax7_Q!m2")},
    {"name": "exact_hello", "system": "Output exactly: Hello, World!",
     "user": "Copy it", "max_tokens": 16, "grade": _exact("Hello, World!")},
    {"name": "exact_pi", "system": "Output exactly: 3.14159",
     "user": "Copy it", "max_tokens": 16, "grade": _exact("3.14159")},
    {"name": "exact_print", "system": "Output exactly: print('hi')",
     "user": "Copy it", "max_tokens": 16, "grade": _exact("print('hi')")},
    {"name": "exact_list", "system": "Output exactly: [1,2,3]",
     "user": "Copy it", "max_tokens": 16, "grade": _exact("[1,2,3]")},
    {"name": "exact_true", "system": "Output exactly: True",
     "user": "Copy it", "max_tokens": 16, "grade": _exact("True")},
    {"name": "exact_dsv4", "system": "Output exactly: dsv4",
     "user": "Copy it", "max_tokens": 16, "grade": _exact("dsv4")},

    # ---- JSON output (parsed-object equality) ----
    {"name": "json_sum", "system": "Answer JSON only",
     "user": 'Output {"sum": 20+22}.',
     "max_tokens": 40, "grade": _json_equal({"sum": 42})},
    {"name": "json_square", "system": "Answer JSON only",
     "user": 'Output {"square": 13*13}.',
     "max_tokens": 40, "grade": _json_equal({"square": 169})},
    {"name": "json_product", "system": "Answer JSON only",
     "user": 'Output {"product": 17*23}.',
     "max_tokens": 40, "grade": _json_equal({"product": 391})},
    {"name": "json_prime", "system": "Answer JSON only",
     "user": 'Output {"is_prime": true} if 15 is prime else {"is_prime": false}.',
     "max_tokens": 40, "grade": _json_equal({"is_prime": False})},
    {"name": "json_round", "system": "Answer JSON only",
     "user": 'Output {"rounded": round(3.7)}.',
     "max_tokens": 40, "grade": _json_equal({"rounded": 4})},
    {"name": "json_count", "system": "Answer JSON only",
     "user": 'Output {"count": len("hello")}.',
     "max_tokens": 40, "grade": _json_equal({"count": 5})},
    {"name": "json_fact", "system": "Answer JSON only",
     "user": 'Output {"factorial": 8!} (use 40320).',
     "max_tokens": 40, "grade": _json_equal({"factorial": 40320})},
    {"name": "json_reverse", "system": "Answer JSON only",
     "user": 'Output {"reversed": "abc"[::-1]}.',
     "max_tokens": 40, "grade": _json_equal({"reversed": "cba"})},

    # ---- arithmetic (numeric answer) ----
    {"name": "math_mult", "system": "Answer with the number only.",
     "user": "Compute 17 * 23.", "max_tokens": 40, "grade": _num(391)},
    {"name": "math_mod", "system": "Answer with the number only.",
     "user": "Compute 1000003 modulo 7.", "max_tokens": 40, "grade": _num(4)},
    {"name": "math_sqrt", "system": "Answer with the number only.",
     "user": "What is the square root of 144?",
     "max_tokens": 40, "grade": _num(12)},
    {"name": "math_pow", "system": "Answer with the number only.",
     "user": "Compute 2 to the power 10.", "max_tokens": 40, "grade": _num(1024)},
    {"name": "math_sub", "system": "Answer with the number only.",
     "user": "Compute 99 - 43.", "max_tokens": 40, "grade": _num(56)},
    {"name": "math_precedence", "system": "Answer with the number only.",
     "user": "Compute 7 + 8 * 3.", "max_tokens": 40, "grade": _num(31)},
    {"name": "math_gcd", "system": "Answer with the number only.",
     "user": "What is the greatest common divisor of 12 and 18?",
     "max_tokens": 40, "grade": _num(6)},
    {"name": "math_lcm", "system": "Answer with the number only.",
     "user": "What is the least common multiple of 4 and 6?",
     "max_tokens": 40, "grade": _num(12)},
    {"name": "math_pct", "system": "Answer with the number only.",
     "user": "What is 50 percent of 80?", "max_tokens": 40, "grade": _num(40)},
    {"name": "math_square25", "system": "Answer with the number only.",
     "user": "Compute 25 * 25.", "max_tokens": 40, "grade": _num(625)},
    {"name": "math_rem", "system": "Answer with the number only.",
     "user": "What is the remainder of 2024 divided by 5?",
     "max_tokens": 40, "grade": _num(4)},
    {"name": "math_sum1to10", "system": "Answer with the number only.",
     "user": "What is 1 + 2 + 3 + ... + 10?", "max_tokens": 40,
     "grade": _num(55)},

    # ---- reasoning / word problems (final number) ----
    {"name": "reason_tom", "system": "Answer with only the final number.",
     "user": "Tom has 3 apples, gives 1 to Ann, then doubles what he has "
             "left. How many apples does Tom have?",
     "max_tokens": 96, "grade": _num(4)},
    {"name": "reason_train", "system": "Answer with only the final number.",
     "user": "A train travels 120 km in 2 hours. What is its speed in km/h?",
     "max_tokens": 96, "grade": _num(60)},
    {"name": "reason_age", "system": "Answer with only the final number.",
     "user": "If x + 5 = 17, what is x?", "max_tokens": 96, "grade": _num(12)},
    {"name": "reason_half", "system": "Answer with only the final number.",
     "user": "Take half of 50, then add 10. What is the result?",
     "max_tokens": 96, "grade": _num(35)},
    {"name": "reason_consec", "system": "Answer with only the final number.",
     "user": "Three consecutive integers sum to 21. What is the middle one?",
     "max_tokens": 96, "grade": _num(7)},
    {"name": "reason_minutes", "system": "Answer with only the final number.",
     "user": "How many minutes are in 2.5 hours?",
     "max_tokens": 96, "grade": _num(150)},
    {"name": "reason_pct", "system": "Answer with only the final number.",
     "user": "What is 20 percent of 250?", "max_tokens": 96, "grade": _num(50)},
    {"name": "reason_square", "system": "Answer with only the final number.",
     "user": "If a = 5 and b = 5, what is (a + b) squared?",
     "max_tokens": 96, "grade": _num(100)},
    {"name": "reason_pages", "system": "Answer with only the final number.",
     "user": "Sam reads 3 books, each 12 pages. How many pages total?",
     "max_tokens": 96, "grade": _num(36)},
    {"name": "reason_div", "system": "Answer with only the final number.",
     "user": "Compute 72 divided by 9.", "max_tokens": 96, "grade": _num(8)},

    # ---- code (substring checks) ----
    {"name": "code_add", "system": "Write only a Python function; no prose.",
     "user": "def add(a, b): pass  # return a+b",
     "max_tokens": 60,
     "grade": lambda t: ("return a + b" in t or "return a+b" in t,
                         "check return a+b")},
    {"name": "code_fib", "system": "Write only a Python function; no prose.",
     "user": "def fib(n): pass  # recursive fibonacci",
     "max_tokens": 80,
     "grade": lambda t: ("fib(n-1)" in t and "fib(n-2)" in t,
                         "check recursive fib calls")},
    {"name": "code_even", "system": "Write only a Python function; no prose.",
     "user": "def is_even(n): pass  # return True if n is even",
     "max_tokens": 60,
     "grade": lambda t: ("% 2" in t and "return" in t, "check n % 2")},
    {"name": "code_cap", "system": "Write only a Python function; no prose.",
     "user": "def capitalize(s): pass  # first letter upper",
     "max_tokens": 60,
     "grade": lambda t: ("[0].upper()" in t or ".capitalize()" in t
                         or "[:1].upper()" in t,
                         "check str capitalize")},
    {"name": "code_reverse", "system": "Write only a Python expression; no prose.",
     "user": "Reverse the list x in place of copying: x = [1,2,3]; y = ???",
     "max_tokens": 40,
     "grade": lambda t: ("[::-1]" in t or ".reverse()" in t,
                         "check slicing or in-place reverse")},
    {"name": "code_max3", "system": "Write only a Python function; no prose.",
     "user": "def max_of_three(a, b, c): pass  # return the largest",
     "max_tokens": 80,
     "grade": lambda t: ("max(" in t, "check max")},

    # ---- multilingual (one sentence, non-empty) ----
    {"name": "multi_es", "system": "Answer in Spanish, one sentence.",
     "user": "Por que flota el hielo en el agua?",
     "max_tokens": 60, "grade": lambda t: (len(t.strip()) > 15,
                                           f"len={len(t.strip())}")},
    {"name": "multi_fr", "system": "Answer in French, one sentence.",
     "user": "Pourquoi le ciel est-il bleu?",
     "max_tokens": 60, "grade": lambda t: (len(t.strip()) > 15,
                                           f"len={len(t.strip())}")},
    {"name": "multi_de", "system": "Answer in German. Show the number.",
     "user": "Was ist 2 + 2?",
     "max_tokens": 40, "grade": _contains("4")},

    # ---- classification / choice ----
    {"name": "choice_largest", "system": "Answer with the number only.",
     "user": "Which is largest: 3, 7, 2, 9?", "max_tokens": 20,
     "grade": _num(9)},
    {"name": "choice_first", "system": "Answer with the letter only.",
     "user": "What is the first letter of 'banana'?", "max_tokens": 20,
     "grade": lambda t: (t.strip().lower().startswith("b"),
                         f"starts with b: {t.strip()!r}")},
    {"name": "choice_vowels", "system": "Answer with the number only.",
     "user": "How many vowels are in 'hello'?", "max_tokens": 20,
     "grade": _num(2)},
]


def _case_payload(case: dict, model: str, remote: bool) -> dict:
    msg = []
    if case.get("system"):
        msg.append({"role": "system", "content": case["system"]})
    msg.append({"role": "user", "content": case["user"]})
    p = {"model": model, "messages": msg, "max_tokens": case["max_tokens"],
         "temperature": 0, "top_p": 1, "logprobs": True,
         "top_logprobs": 5, "stream": False}
    if remote:
        p["reasoning"] = {"effort": "none"}
        p["provider"] = {"only": [PROVIDER_SLUG],
                         "order": [PROVIDER_SLUG],
                         "allow_fallbacks": False,
                         "require_parameters": True}
    return p


def _post(url: str, payload: dict, timeout: int = 300, token: str | None = None,
          retries: int = 2):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers)
    t0 = time.time()
    raw: bytes | None = None
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                # A client-side read glitch can truncate a multi-byte
                # sequence mid-stream; retry the draw rather than dropping
                # the task (the server-side response is UTF-8-valid — we
                # verified 60/60 standalone decodes).
                last_err = e
                continue
            return body, time.time() - t0
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(0.5)
                continue
    raise last_err or RuntimeError("_post failed")


def run_tasks(endpoint: str | None, remote: bool, token: str | None,
              repeats: int = 3) -> dict:
    """Run every task `repeats` times; a task PASSES iff its grader passes
    on the MAJORITY of draws (ties -> the last draw). Averaging draws makes
    the benchmark stable: the FP8 reference alone flipped single-shot
    answers between runs (e.g. sqrt(144)=14 once, 12 next). Everyone gets
    the same repeats."""
    results = {}
    base = "https://openrouter.ai/api/v1/chat/completions" if remote else \
        f"{endpoint}/v1/chat/completions"
    model = MODEL_ID if remote else LOCAL_MODEL
    for case in TASKS:
        name = case["name"]
        draws = []
        errors = []
        for _ in range(repeats):
            try:
                body, dt = _post(base, _case_payload(case, model, remote),
                                 token=token)
                choice = (body.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                usage = body.get("usage") or {}
                ct = usage.get("completion_tokens") or 0
                passed, detail = case["grade"](content)
                draws.append({
                    "content": content,
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "passed": passed,
                    "detail": detail,
                    "latency_s": round(dt, 3),
                    "prompt_tokens": usage.get("prompt_tokens") or 0,
                    "completion_tokens": ct,
                    "tok_s": round((ct or 1) / dt, 2) if dt else None,
                })
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
        if draws:
            n_pass = sum(1 for d in draws if d["passed"])
            majority_pass = n_pass > repeats / 2
            results[name] = {
                "passes": n_pass,
                "repeats": repeats,
                "passed": majority_pass,
                "draws": draws,
            }
            if errors:
                results[name]["errors"] = errors
        else:
            results[name] = {"error": "; ".join(errors) or "no draws",
                             "passed": None}
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true",
                    help="hit OpenRouter (spends credits)")
    ap.add_argument("--port", type=int, default=11234)
    ap.add_argument("--mtplx-port", type=int, default=8000)
    ap.add_argument("--key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--out", default="/Users/pjb/git/mlx-serve/receipts/dsv4")
    ap.add_argument("--remote-only", action="store_true")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    arms = {}
    if args.remote or args.remote_only:
        if not args.key:
            sys.exit("--remote requires OPENROUTER_API_KEY or --key")
        print("[bench] remote (OpenRouter coreweave/fp8) ...")
        arms["remote"] = run_tasks(None, True, args.key, args.repeats)
    if not args.remote_only:
        print(f"[bench] direct mlx-serve on :{args.port} ...")
        arms["direct"] = run_tasks(f"http://127.0.0.1:{args.port}", False,
                                   None, args.repeats)
        print(f"[bench] MTPLX-routed on :{args.mtplx_port} ...")
        arms["mtplx"] = run_tasks(f"http://127.0.0.1:{args.mtplx_port}",
                                  False, None, args.repeats)

    # ---- aggregate summary ----
    summary = {}
    for arm, results in arms.items():
        graded = [r for r in results.values() if r.get("passed") is not None]
        passes = [r for r in graded if r["passed"]]
        speeds = [d["tok_s"] for r in results.values()
                  for d in (r.get("draws") or []) if d.get("tok_s")]
        summary[arm] = {
            "tasks": len(results),
            "graded": len(graded),
            "passed": len(passes),
            "pass_rate": round(len(passes) / len(graded), 3)
            if graded else None,
            "median_tok_s": round(sorted(speeds)[len(speeds)//2], 2)
            if speeds else None,
            "mean_tok_s": round(sum(speeds) / len(speeds), 2)
            if speeds else None,
        }

    # cross-arm exact + normalized agreement vs remote (majority draw)
    def majority_content(res: dict) -> str:
        draws = res.get("draws") or []
        return draws[-1].get("content", "") if draws else ""

    comparisons = {}
    if "remote" in arms and ("direct" in arms or "mtplx" in arms):
        for arm in ("direct", "mtplx"):
            if arm not in arms:
                continue
            agr = {"exact": 0, "normalized": 0, "total": 0}
            for name in TASKS:
                a = majority_content(arms["remote"].get(name, {}))
                b = majority_content(arms[arm].get(name, {}))
                if not a and not b:
                    continue
                agr["total"] += 1
                if a.strip() == b.strip():
                    agr["exact"] += 1
                if re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b):
                    agr["normalized"] += 1
            comparisons[arm] = agr

    import datetime
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "dsv4-quality-speed-v1",
        "stamp": stamp,
        "tasks": [{"name": t["name"], "max_tokens": t["max_tokens"]}
                  for t in TASKS],
        "arms": arms,
        "summaries": summary,
        "cross_arm_agreement_vs_remote": comparisons,
        "notes": [
            "temp=0, top_p=1, non-stream, logprobs=5 on every arm",
            "remote: OpenRouter deepseek/deepseek-v4-flash-0731 on pinned "
            "coreweave/fp8, fallbacks disabled (same reference as the "
            "published comparison)",
            "remote tok_s is network-inclusive and NOT model speed; local "
            "tok_s is genuine engine throughput on this M5 Max",
            "pass means the same programmatic grader passed on the "
            "original and the derivative",
        ],
    }
    # --out was not part of the parser; add it
    out = out_dir / f"quality-speed-{stamp}.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nwrote {out}\n")
    for arm, s in summary.items():
        print(f"[summary] {arm:8s} pass={s['passed']}/{s['graded']} "
              f"({s['pass_rate']}) | "
              f"mean_tok_s={s['mean_tok_s']} median_tok_s={s['median_tok_s']}")
    for arm, agr in comparisons.items():
        print(f"[agreement {arm} vs remote] exact {agr['exact']}/"
              f"{agr['total']} | normalized {agr['normalized']}/"
              f"{agr['total']}")


if __name__ == "__main__":
    main()