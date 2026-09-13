#!/usr/bin/env python3
"""Summarize every report in a completed PR #302 session, without exclusions."""
import argparse
import json
from pathlib import Path
from statistics import median

from bench_pr302_abba import ARMS, RUNGS, digest, validate_report, write_json


def stats(values):
    return {"median": median(values), "min": min(values), "max": max(values), "samples": values}


def cell(s):
    return f"{s['median']:.1f} ({s['min']:.1f}–{s['max']:.1f})"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session", type=Path)
    a = p.parse_args()
    m = json.loads((a.session / "manifest.json").read_text())
    if not m.get("completed") or [r["arm"] for r in m["runs"]] != list(ARMS):
        raise SystemExit("refusing to summarize an incomplete or reordered session")
    runs = []
    for r in m["runs"]:
        stem = f"{r['ordinal']:02d}-{r['arm']}"
        path = a.session / f"{stem}.llmprobe.json"
        if digest(path) != r["report_sha256"]:
            raise SystemExit(f"report changed: {path}")
        d = json.loads(path.read_text())
        validate_report(d, (a.session / f"{stem}.server.log").read_text(), r["arm"])
        runs.append({"run": stem, "arm": r["arm"], "bench": d["bench"]})
    summary = {"short_prompt": {}, "context": {}, "runs": runs}
    short = {"decode": lambda b: b["decodeTokPerSec"]["median"],
             "novel_spec": lambda b: b["speculative"]["novelTokPerSec"],
             "predictable_spec": lambda b: b["speculative"]["predictableTokPerSec"],
             "prefill": lambda b: b["prefillTokPerSec"]["median"],
             "ttft_ms": lambda b: b["ttftMs"]["median"]}
    for name, extract in short.items():
        summary["short_prompt"][name] = {
            arm: stats([extract(r["bench"]) for r in runs if r["arm"] == arm]) for arm in "AB"}
    for idx, target in enumerate(RUNGS):
        metrics = {}
        for name in ("decodeTokPerSec", "prefillTokPerSec", "ttftMs"):
            groups = {arm: stats([r["bench"]["contextScaling"][idx][name]
                                 for r in runs if r["arm"] == arm]) for arm in "AB"}
            groups["median_change_pct"] = 100 * (groups["B"]["median"] / groups["A"]["median"] - 1)
            metrics[name] = groups
        summary["context"][str(target)] = metrics
    write_json(a.session / "summary.json", summary)
    lines = ["## Measured results", "", "Every value below retains all four reports per arm. "
             "Aggregate cells are median (minimum–maximum) across report values; these ranges are not confidence intervals.", "",
             "| Context | Control A novel decode tok/s | PR B novel decode tok/s | Median change | A prefill tok/s | B prefill tok/s |",
             "|---|---:|---:|---:|---:|---:|"]
    for target in RUNGS:
        c = summary["context"][str(target)]
        d, f = c["decodeTokPerSec"], c["prefillTokPerSec"]
        lines.append(f"| {target // 1024}K | {cell(d['A'])} | {cell(d['B'])} | "
                     f"{d['median_change_pct']:+.1f}% | {cell(f['A'])} | {cell(f['B'])} |")
    lines += ["", "| Short-prompt metric | Control A | PR B |", "|---|---:|---:|"]
    for name, s in summary["short_prompt"].items():
        lines.append(f"| {name} | {cell(s['A'])} | {cell(s['B'])} |")
    lines += ["", "## Individual reports", "", "| Run | Novel spec tok/s | 2K novel tok/s | 8K | 16K | 32K | Load drift |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for r in runs:
        b = r["bench"]
        row = [f"[{r['run']}]({r['run']}.llmprobe.json)", str(b["speculative"]["novelTokPerSec"])]
        row += [str(x["decodeTokPerSec"]) for x in b["contextScaling"]]
        row += [f"{b['loadDrift']['driftPct']:+.1f}% ({b['loadDrift']['verdict']})"]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Engagement", "", "| Run | Width-9 split lines | Depth-8 MTP trace lines |",
              "|---|---:|---:|"]
    for r in m["runs"]:
        e = r["engagement"]
        lines.append(f"| {r['ordinal']:02d}-{r['arm']} | {e['split_q9_count']} | {e['mtp_depth8_trace_count']} |")
    (a.session / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
