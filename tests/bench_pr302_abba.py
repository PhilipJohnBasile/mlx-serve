#!/usr/bin/env python3
"""PR #302: one-boot ABBA/ABBA llmprobe comparison with auditable artifacts.

Set Macs Fan Control to Full blast before starting; retain that preset until
completion. Both binaries must be freshly built with the same dependencies.
The output directory must be new. No existing server is terminated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

ARMS = tuple("ABBAABBA")
RUNGS = [2048, 8192, 16384, 32768]
CTX_SIZE = 40960  # Template/calibration overshoot plus generated tokens at 32K.
PROBE_VERSION = "0.6.7"  # 0.6.1 rejects a 2K rung.
ENV = {"MLX_SERVE_MTP_FORCE_DEPTH": "8", "MLX_SERVE_MTP_TRACE": "1"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def port_free():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", 11234)) != 0


def validate_report(report, log, arm):
    """Fail closed on missing/failed rungs and an unproven treatment arm."""
    if report["run"]["mode"] != "bench-only":
        raise ValueError("not a benchmark-only report")
    points = report["bench"]["contextScaling"]
    if [p["targetTokens"] for p in points] != RUNGS:
        raise ValueError("missing or incorrect context rungs")
    for p in points:
        if p["runs"] < 1 or p["note"] or any(
            p[k] is None or p[k] <= 0
            for k in ("inputTokens", "decodeTokPerSec", "ttftMs", "prefillTokPerSec")
        ):
            raise ValueError(f"failed context rung: {p}")
        if abs(p["inputTokens"] / p["targetTokens"] - 1) > 0.15:
            raise ValueError(f"context calibration missed target: {p}")
    split9 = re.findall(r"[^\n]*\[sdpa-split\] engaged: qL=9\b[^\n]*", log)
    trace8 = re.findall(r"[^\n]*\[mtp-trace\][^\n]*m_avg=8\.00\b[^\n]*", log)
    nax = re.findall(r"[^\n]*\[nax-sdpa\] engaged:[^\n]*", log)
    if (arm == "A") != bool(split9):
        raise ValueError(f"arm {arm}: unexpected width-9 split engagement")
    if not trace8 or not nax or "[spec-stats] mode=mtp" not in log:
        raise ValueError("missing positive forced-depth-8 MTP / NAX evidence")
    return {"split_q9_count": len(split9), "split_q9_lines": split9,
            "mtp_depth8_trace_count": len(trace8), "mtp_depth8_example": trace8[:1],
            "nax_lines": nax}


def post(model, content):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "temperature": 0, "max_tokens": 192, "min_tokens": 192,
                       "ignore_eos": True, "enable_mtp": True, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:11234/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fans-note", required=True, help="Observed Full blast RPMs and time")
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    bins = {"A": a.control.resolve(), "B": a.candidate.resolve()}
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_")}
    env.update(ENV)
    boot = command("sysctl", "-n", "kern.boottime")
    power = command("pmset", "-g", "batt")
    if "AC Power" not in power or not port_free():
        raise RuntimeError("requires AC power and an unused port 11234")
    probe = ["npx", "--yes", f"llmprobe@{PROBE_VERSION}"]
    if command(*probe, "--version") != PROBE_VERSION:
        raise RuntimeError("unexpected llmprobe version")
    manifest = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "boot": boot, "os": command("sw_vers"), "power": power,
                "fans_note": a.fans_note, "sequence": list(ARMS), "rungs": RUNGS,
                "context_limit": CTX_SIZE, "env": ENV, "llmprobe_version": PROBE_VERSION,
                "model": str(a.model), "model_files": {}, "arms": {}, "runs": []}
    for f in sorted(a.model.iterdir()):
        if f.is_file():
            manifest["model_files"][f.name] = {"bytes": f.stat().st_size, "sha256": digest(f)}
    for arm, binary in bins.items():
        root = binary.parent.parent.parent
        manifest["arms"][arm] = {"binary": str(binary), "sha256": digest(binary),
                                  "commit": command("git", "-C", str(root), "rev-parse", "HEAD"),
                                  "version": command(str(binary), "--version"), "libraries": {}}
        for f in sorted((root / "lib/mlx/lib").glob("*")):
            if f.is_file():
                manifest["arms"][arm]["libraries"][f.name] = digest(f)
    if manifest["arms"]["A"]["libraries"] != manifest["arms"]["B"]["libraries"]:
        raise RuntimeError("arms must use identical MLX libraries")
    write_json(a.output / "manifest.json", manifest)
    for i, arm in enumerate(ARMS, 1):
        if command("sysctl", "-n", "kern.boottime") != boot:
            raise RuntimeError("boot changed; start a new session")
        power = command("pmset", "-g", "batt")
        if "AC Power" not in power or not port_free():
            raise RuntimeError("AC power lost or port already occupied")
        stem = f"{i:02d}-{arm}"
        log_path = a.output / f"{stem}.server.log"
        report_path = a.output / f"{stem}.llmprobe.json"
        argv = [str(bins[arm]), "--serve", "--host", "127.0.0.1", "--port", "11234",
                "--model", str(a.model), "--no-vision", "--ctx-size", str(CTX_SIZE),
                "--mtp", "--mtp-depth", "8", "--no-pld", "--prefix-cache-entries", "0",
                "--log-level", "debug", "--log-file", str(log_path)]
        row = {"ordinal": i, "arm": arm, "server_command": argv, "power_before": power,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(f"{stem}: starting server; fixed warmups then four context rungs", flush=True)
        with (a.output / f"{stem}.console.log").open("w") as console:
            server = subprocess.Popen(argv, env=env, cwd=bins[arm].parent.parent.parent,
                                      stdout=console, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 240
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited {server.returncode}")
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:11234/v1/models", timeout=2) as r:
                            models = json.load(r)["data"]
                        if models:
                            model_id = models[0]["id"]
                            break
                    except (OSError, ValueError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("server readiness")
                    time.sleep(1)
                # Equal fixed work before every measured run. These are discarded
                # warmups, saved separately and never included in throughput means.
                for w, size in enumerate((2000, 30000), 1):
                    content = "\n".join(f"Record {j}: the station logged wind, cloud and rainfall."
                                        for j in range(size // 13))
                    content += "\nWrite a detailed original analysis of this weather archive."
                    response = post(model_id, content)
                    write_json(a.output / f"{stem}.warmup-{w}.json", response)
                probe_cmd = probe + ["localhost:11234", "--bench-only", "--rungs", "2k,8k,16k,32k",
                                     "--no-save", "--no-color", "--save", str(report_path)]
                row["probe_command"] = probe_cmd
                print(f"{stem}: running llmprobe", flush=True)
                with (a.output / f"{stem}.llmprobe.log").open("w") as probe_log:
                    subprocess.run(probe_cmd, stdout=probe_log, stderr=subprocess.STDOUT,
                                   check=True, timeout=1800, env=env)
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
        report = json.loads(report_path.read_text())
        row["engagement"] = validate_report(report, log_path.read_text(), arm)
        row["boot_after"] = command("sysctl", "-n", "kern.boottime")
        row["power_after"] = command("pmset", "-g", "batt")
        if row["boot_after"] != boot or "AC Power" not in row["power_after"]:
            raise RuntimeError("boot or power changed during run")
        row["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        row["report_sha256"] = digest(report_path)
        manifest["runs"].append(row)
        write_json(a.output / "manifest.json", manifest)
        print(f"{stem}: PASS all four rungs and forced-depth engagement; "
              f"drift={report['bench']['loadDrift']}", flush=True)
    manifest["completed"] = True
    write_json(a.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
