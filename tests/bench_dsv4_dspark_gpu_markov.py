#!/usr/bin/env python3
"""Proof-grade local benchmark for DSV4 speculative-decoding fast paths.

This intentionally compares separate fresh server boots of the same
ReleaseFast binary and model directory:

  S  plain serial (no ``--dspark``)
  L  legacy DSpark (``--dspark``)
  G  DSpark plus ``MLX_SERVE_DSPARK_GPU_MARKOV_IDS=1``
  J  G plus ``MLX_SERVE_DSPARK_JOIN_VERIFY_EVAL=1``
  B2 GPU-Markov DSpark, ``MLX_SERVE_DSPARK_BLOCK_CAP=2``, fast defaults
  R2 B2 plus exact ``MLX_SERVE_DSPARK_REPLAY_COMMIT=1``

Every boot receives one fixed, excluded warmup followed by the measured
temperature-0 prompt (64 tokens by default).  It writes the server log, request and
metrics JSON, sanitized environment, exact argv, and binary/model identities
to a new receipt directory.  It refuses to report a speed comparison unless
all measured output contracts have the same SHA-256 hash.

The default ``pilot`` is a one-pass S/L/G screen (one measured request per
boot).  ``--mode balanced`` performs six counterbalanced S/L/G orders
SLG,LGS,GSL,GLS,SGL,LSG, with one measured request per fresh boot.  Selecting
all four arms uses the frozen position-balanced orders
SLGJ,LGJS,GJSL,JSLG.  This is deliberately serial: the full model must never
be loaded twice at once on the benchmark machine.

The explicit ``--execution-profile fast-default --arms S,B2`` experiment
compares a fast-default serial baseline to the capped B2 DSpark lane.  Its
balanced schedule is S,B2 then B2,S, so every arm occupies each position once.
The replay comparison is ``--execution-profile fast-default --arms S,B2,R2``;
its balanced schedule runs the forward/reverse S,B2,R2 and R2,B2,S orders.
All inherited ``MLX_SERVE_*`` variables are still removed; "fast default"
means the three DSV4 kill switches are deliberately absent, not inherited.

``--arms G`` (or ``--arms S,J``) is a recovery/pilot mode for selected arms.
It preserves the selected part of the requested schedule and clearly marks
the resulting receipt as incomplete for cross-arm speed comparison.  Pass
``--compare-receipt /path/to/prior-receipt`` to import completed measurements
from a named compatible receipt and make the exact-output gate span both
receipts.  A prior receipt proves output equality, not same-run timing.

Examples (build ReleaseFast first):

  python3 tests/bench_dsv4_dspark_gpu_markov.py --mode pilot
  python3 tests/bench_dsv4_dspark_gpu_markov.py --mode balanced
  python3 tests/bench_dsv4_dspark_gpu_markov.py --mode pilot --arms S,J --max-tokens 16
  python3 tests/bench_dsv4_dspark_gpu_markov.py --execution-profile fast-default --arms S,B2 --mode pilot --max-tokens 8
  python3 tests/bench_dsv4_dspark_gpu_markov.py --execution-profile fast-default --arms S,B2,R2 --mode balanced --max-tokens 8

``--self-test`` is offline and exercises only the harness's schedule,
metrics-delta, environment-sanitizing, and log-evidence logic.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


MANIFEST_SCHEMA = "mlx-serve.dsv4-dspark-fast-path-benchmark"
MANIFEST_VERSION = 4
DEFAULT_ARMS = ("S", "L", "G")
LEGACY_ALL_ARMS = ("S", "L", "G", "J")
ALL_ARMS = (*LEGACY_ALL_ARMS, "B2", "R2")
B2_ARMS = ("S", "B2")
R2_ARMS = ("S", "B2", "R2")
BALANCED_ORDERS = (
    ("S", "L", "G"),
    ("L", "G", "S"),
    ("G", "S", "L"),
    ("G", "L", "S"),
    ("S", "G", "L"),
    ("L", "S", "G"),
)
FOUR_ARM_BALANCED_ORDERS = (
    ("S", "L", "G", "J"),
    ("L", "G", "J", "S"),
    ("G", "J", "S", "L"),
    ("J", "S", "L", "G"),
)
B2_BALANCED_ORDERS = (
    ("S", "B2"),
    ("B2", "S"),
)
R2_BALANCED_ORDERS = (
    ("S", "B2", "R2"),
    ("R2", "B2", "S"),
)
ARM_NAMES = {
    "S": "plain_serial",
    "L": "legacy_dspark",
    "G": "dspark_gpu_markov_ids",
    "J": "dspark_gpu_markov_ids_joined_verify_eval",
    "B2": "dspark_gpu_markov_ids_block_cap_2_fast_defaults",
    "R2": "dspark_gpu_markov_ids_block_cap_2_replay_commit_fast_defaults",
}

# Deliberately remove every MLX_SERVE_* inherited setting.  A benchmark must
# not quietly inherit an earlier experiment's verifier, tracing, quantization,
# or speculative-mode switch. These values are the complete common
# conservative configuration; G and J add only their documented fast-path
# switches.
COMMON_ENV = {
    "MLX_SERVE_DSV4_DEC_CHAIN": "0",
    "MLX_SERVE_DSV4_MOE_ROUTE_GPU": "0",
    "MLX_SERVE_DSV4_LAZY_DECODE": "0",
}
FAST_DEFAULT_ENV: dict[str, str] = {}
EXECUTION_PROFILES = {
    "conservative": {
        "common_mlx_serve_env": COMMON_ENV,
        "resolved_source_defaults": {
            "MLX_SERVE_DSV4_DEC_CHAIN": "explicitly disabled",
            "MLX_SERVE_DSV4_MOE_ROUTE_GPU": "explicitly disabled",
            "MLX_SERVE_DSV4_LAZY_DECODE": "explicitly disabled",
        },
    },
    "fast-default": {
        "common_mlx_serve_env": FAST_DEFAULT_ENV,
        "resolved_source_defaults": {
            "MLX_SERVE_DSV4_DEC_CHAIN": "unset => enabled",
            "MLX_SERVE_DSV4_MOE_ROUTE_GPU": "unset => enabled",
            "MLX_SERVE_DSV4_LAZY_DECODE": "unset => enabled",
        },
    },
}
GPU_MARKOV_ENV = "MLX_SERVE_DSPARK_GPU_MARKOV_IDS"
JOIN_VERIFY_ENV = "MLX_SERVE_DSPARK_JOIN_VERIFY_EVAL"
BLOCK_CAP_ENV = "MLX_SERVE_DSPARK_BLOCK_CAP"
REPLAY_COMMIT_ENV = "MLX_SERVE_DSPARK_REPLAY_COMMIT"
WIRED_ENV = "MLX_SERVE_WIRED"
DSPARK_PROFILE_ENV = "MLX_SERVE_DSPARK_PROFILE"
DSPARK_PROFILE_EVERY_ENV = "MLX_SERVE_DSPARK_PROFILE_EVERY"
GPU_MARKOV_MARKER = (
    "dsv4: DSpark GPU Markov-ID path engaged "
    "(one typed ID/confidence eval; draft logits stay on device)"
)
JOIN_VERIFY_MARKER = "dsv4: DSpark joined verify/deferred-row eval engaged (one GPU barrier)"
BLOCK_CAP_B2_MARKER = "dsv4: DSpark block cap engaged (effective=2,"
REPLAY_COMMIT_MARKER = (
    "dsv4: DSpark replay-commit verifier engaged "
    "(GPU emissions only; retained prefix serially committed)"
)
DSPARK_STATS_MARKER = "[spec-stats] mode=dspark"
DSPARK_PROFILE_MARKER = "[dspark-prof]"
LOWER_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_PROMPT = (
    "You are reviewing a local inference engine. In four concise bullets, "
    "explain how to improve speculative decoding while preserving exact "
    "greedy-token equivalence. Cover KV-cache ownership, batched "
    "verification, GPU/host synchronization, and regression tests."
)


class BenchError(RuntimeError):
    """A receipt-preserving benchmark failure."""


@dataclass
class ServerHandle:
    process: subprocess.Popen[bytes]
    log_path: Path
    stream: Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proof-grade local DSV4 DSpark fast-path benchmark."
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("./zig-out/bin/mlx-serve"),
        help="ReleaseFast mlx-serve binary (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/tmp/dspark-lite-A"),
        help="same DSV4 model directory for every arm (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "balanced"),
        default="pilot",
        help=(
            "pilot=one pass; balanced=six S/L/G counterbalanced orders or "
            "four position-balanced orders when J is selected; fast-default "
            "S,B2 uses its own two-order position-balanced schedule, while "
            "S,B2,R2 uses forward/reverse S,B2,R2 and R2,B2,S orders"
        ),
    )
    parser.add_argument(
        "--execution-profile",
        choices=tuple(EXECUTION_PROFILES),
        default="conservative",
        help=(
            "conservative disables three DSV4 fast paths (default); "
            "fast-default is permitted only for explicit S,B2 or S,B2,R2 "
            "experiments and leaves those kill switches absent after sanitization"
        ),
    )
    parser.add_argument(
        "--wired-mode",
        choices=("default", "off"),
        default="default",
        help=(
            "common wired-kernel policy for every arm: default leaves "
            "MLX_SERVE_WIRED unset after sanitization; off sets it to off "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        metavar="S,L,G,J,B2,R2",
        help=(
            "selected/recovery subset, e.g. --arms G or --arms S,J; "
            "S,B2 is the fast-default B2 experiment and S,B2,R2 is its "
            "replay-commit extension; default is S,L,G"
        ),
    )
    parser.add_argument(
        "--dspark-profile-every",
        type=int,
        metavar="ROUNDS",
        help=(
            "B2 pilot-only diagnostic: enable DSpark phase profiling and report "
            "every ROUNDS (1..1024); profiled timing is explicitly diagnostic"
        ),
    )
    parser.add_argument(
        "--compare-receipt",
        type=Path,
        help=(
            "prior compatible receipt directory; imports completed measurements "
            "into the exact-output gate only"
        ),
    )
    parser.add_argument(
        "--measured-per-boot",
        type=int,
        default=None,
        help="measured requests after the excluded warmup (must be exactly 1)",
    )
    parser.add_argument(
        "--warmups-per-boot",
        type=int,
        default=1,
        help="fixed excluded warmups per fresh boot (must be exactly 1)",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=None,
        help="first loopback port; each boot receives a unique following port",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for a model server to become healthy",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=900.0,
        help="seconds allowed for each completion request",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="completion-token cap for warmups and measurements (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="new receipt directory (default: receipts/dsv4/...timestamp...)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="fixed benchmark prompt used for all warmups and measurements",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="read the fixed benchmark prompt from this UTF-8 file",
    )
    parser.add_argument(
        "--model-hash-mode",
        choices=("metadata", "full"),
        default="metadata",
        help=(
            "metadata hashes paths/sizes/mtimes plus small configs without reading "
            "all weights; full SHA-256 hashes every model file before serving"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="offline harness checks; does not inspect paths or start a server",
    )
    return parser.parse_args(argv)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_contract_sha256(contract: dict[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def verified_measurement_hash(measurement: dict[str, Any], context: str) -> str:
    """Recompute, syntax-check, and authenticate an embedded output contract.

    Legacy receipts that stored only a claimed digest are intentionally not
    proof-grade. Raw-response fallback is omitted: receipt paths may be moved
    or attacker-controlled, while the embedded contract is sufficient and
    unambiguous.
    """
    claimed = measurement.get("output_contract_sha256")
    if not isinstance(claimed, str) or LOWER_HEX_SHA256.fullmatch(claimed) is None:
        raise BenchError(f"{context} has no valid 64-character lowercase SHA-256")
    contract = measurement.get("output_contract")
    if not isinstance(contract, dict):
        raise BenchError(
            f"{context} lacks embedded output_contract; legacy hash-only receipts "
            "are not proof-grade"
        )
    if set(contract) != {"content", "finish_reason", "completion_tokens"}:
        raise BenchError(f"{context} output_contract has an unexpected schema")
    if not isinstance(contract["content"], str):
        raise BenchError(f"{context} output_contract.content is not text")
    if contract["finish_reason"] is not None and not isinstance(
        contract["finish_reason"], str
    ):
        raise BenchError(f"{context} output_contract.finish_reason is invalid")
    completion_tokens = contract["completion_tokens"]
    if completion_tokens is not None and (
        isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int)
    ):
        raise BenchError(f"{context} output_contract.completion_tokens is invalid")
    computed = output_contract_sha256(contract)
    if computed != claimed:
        raise BenchError(
            f"{context} output_contract SHA-256 mismatch: claimed {claimed}, computed {computed}"
        )
    return computed


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now_label() -> str:
    return time.strftime("%Y-%m-%d-%H%M%S", time.localtime())


def schedule_for(
    mode: str, selected_arms: tuple[str, ...] = DEFAULT_ARMS
) -> tuple[tuple[str, ...], ...]:
    if mode == "pilot":
        return (ALL_ARMS,)
    if mode == "balanced":
        if set(selected_arms) == set(R2_ARMS):
            return R2_BALANCED_ORDERS
        if set(selected_arms) == set(B2_ARMS):
            return B2_BALANCED_ORDERS
        if "J" in selected_arms:
            return FOUR_ARM_BALANCED_ORDERS
        return BALANCED_ORDERS
    raise ValueError(f"unknown mode {mode!r}")


def parse_selected_arms(raw_values: list[str] | None) -> tuple[str, ...]:
    """Accept ``G``, ``S,L,G``, or ``S L G`` without changing default order."""
    if raw_values is None:
        return DEFAULT_ARMS
    selected: list[str] = []
    for raw in raw_values:
        for arm in raw.split(","):
            arm = arm.strip().upper()
            if not arm:
                continue
            if arm not in ARM_NAMES:
                raise BenchError(
                    f"--arms accepts only S, L, G, J, B2, and R2; got {arm!r}"
                )
            if arm in selected:
                raise BenchError(f"--arms contains duplicate arm {arm!r}")
            selected.append(arm)
    if not selected:
        raise BenchError("--arms selected no arms")
    return tuple(selected)


def filtered_schedule(
    schedule: tuple[tuple[str, ...], ...], selected_arms: tuple[str, ...]
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Keep source-order metadata while assigning ports only to selected boots."""
    filtered = []
    selected_set = set(selected_arms)
    for order in schedule:
        effective = tuple(arm for arm in order if arm in selected_set)
        if effective:
            filtered.append((order, effective))
    if not filtered:
        raise BenchError("the selected arms do not occur in this benchmark schedule")
    return tuple(filtered)


def profile_provenance(execution_profile: str) -> dict[str, Any]:
    if execution_profile not in EXECUTION_PROFILES:
        raise ValueError(f"unknown execution profile {execution_profile!r}")
    profile = EXECUTION_PROFILES[execution_profile]
    return {
        "name": execution_profile,
        "common_mlx_serve_env": dict(profile["common_mlx_serve_env"]),
        "resolved_source_defaults": dict(profile["resolved_source_defaults"]),
        "sanitization": "all inherited MLX_SERVE_* variables removed before each boot",
    }


def b2_extra_env(dspark_profile_every: int | None) -> dict[str, str]:
    env = {
        GPU_MARKOV_ENV: "1",
        BLOCK_CAP_ENV: "2",
    }
    if dspark_profile_every is not None:
        env[DSPARK_PROFILE_ENV] = "1"
        env[DSPARK_PROFILE_EVERY_ENV] = str(dspark_profile_every)
    return env


def r2_extra_env() -> dict[str, str]:
    """R2 is B2 with replay commit enabled by an exact, sanitized opt-in."""
    return {
        **b2_extra_env(None),
        REPLAY_COMMIT_ENV: "1",
    }


def wired_policy_provenance(wired_mode: str) -> dict[str, Any]:
    if wired_mode == "default":
        return {
            "mode": "default",
            "common_mlx_serve_env": {},
            "effective_policy": f"{WIRED_ENV} unset => source default",
        }
    if wired_mode == "off":
        return {
            "mode": "off",
            "common_mlx_serve_env": {WIRED_ENV: "off"},
            "effective_policy": f"{WIRED_ENV}=off for every arm",
        }
    raise ValueError(f"unknown wired mode {wired_mode!r}")


def sanitized_server_env(
    arm: str,
    execution_profile: str = "conservative",
    dspark_profile_every: int | None = None,
    wired_mode: str = "default",
) -> tuple[dict[str, str], list[str]]:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}")
    if execution_profile not in EXECUTION_PROFILES:
        raise ValueError(f"unknown execution profile {execution_profile!r}")
    inherited = dict(os.environ)
    removed = sorted(key for key in inherited if key.startswith("MLX_SERVE_"))
    env = {key: value for key, value in inherited.items() if not key.startswith("MLX_SERVE_")}
    env.update(EXECUTION_PROFILES[execution_profile]["common_mlx_serve_env"])
    env.update(wired_policy_provenance(wired_mode)["common_mlx_serve_env"])
    if arm in ("G", "J"):
        env[GPU_MARKOV_ENV] = "1"
    if arm == "J":
        env[JOIN_VERIFY_ENV] = "1"
    if arm == "B2":
        env.update(b2_extra_env(dspark_profile_every))
    if arm == "R2":
        env.update(r2_extra_env())
    return env, removed


def common_argv(binary: Path, model: Path, port: int, arm: str) -> list[str]:
    argv = [
        str(binary),
        "--model",
        str(model),
        "--serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "12288",
        "--timeout",
        "0",
        "--no-pld",
        "--kv-quant",
        "off",
        "--no-decode-attn-quant",
        "--prefix-cache-entries",
        "0",
        "--prefix-cache-mem",
        "0",
        "--prefix-cache-disk",
        "0",
        "--tokenize-cache-entries",
        "0",
        "--skip-mem-preflight",
        "--metrics",
    ]
    if arm != "S":
        argv.append("--dspark")
    return argv


def request_body(model: Path, prompt: str, arm: str, max_tokens: int) -> dict[str, Any]:
    # ``enable_mtp`` is the per-request DSpark intent for a stage-bearing DSV4.
    # S explicitly opts out, while L/G/J explicitly opt in rather than relying on
    # a server-default change.  PLD and external drafters stay disabled.
    return {
        "model": model.name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "enable_pld": False,
        "enable_drafter": False,
        "enable_mtp": arm != "S",
    }


def effective_request_config(model: Path, max_tokens: int) -> dict[str, Any]:
    """Output-affecting request settings shared by/import-compatible receipts."""
    return {
        "endpoint": "/v1/chat/completions",
        "model": model.name,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "enable_pld": False,
        "enable_drafter": False,
        "enable_mtp_by_arm": {arm: arm != "S" for arm in ALL_ARMS},
    }


def manifest_provenance(
    harness_path: Path,
    startup_timeout: float,
    request_timeout: float,
    request_config: dict[str, Any],
) -> dict[str, Any]:
    harness_path = harness_path.resolve()
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "harness": {
            "path": str(harness_path),
            "sha256": sha256_file(harness_path),
        },
        "timeouts_seconds": {
            "startup": startup_timeout,
            "request": request_timeout,
        },
        "effective_request_config": request_config,
    }


def effective_arm_config(dspark_profile_every: int | None = None) -> dict[str, Any]:
    return {
        "S": {"dspark_server_flag": False, "extra_mlx_serve_env": {}},
        "L": {"dspark_server_flag": True, "extra_mlx_serve_env": {}},
        "G": {
            "dspark_server_flag": True,
            "extra_mlx_serve_env": {GPU_MARKOV_ENV: "1"},
        },
        "J": {
            "dspark_server_flag": True,
            "extra_mlx_serve_env": {
                GPU_MARKOV_ENV: "1",
                JOIN_VERIFY_ENV: "1",
            },
        },
        "B2": {
            "dspark_server_flag": True,
            "extra_mlx_serve_env": b2_extra_env(dspark_profile_every),
        },
        "R2": {
            "dspark_server_flag": True,
            "extra_mlx_serve_env": r2_extra_env(),
        },
    }


def http_json(url: str, *, payload: dict[str, Any] | None = None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise BenchError(f"HTTP {error.code} from {url}: {body[:1000]}") from error
    except urllib.error.URLError as error:
        raise BenchError(f"network error from {url}: {error.reason}") from error
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        excerpt = raw[:1000].decode("utf-8", "replace")
        raise BenchError(f"non-JSON response from {url}: {excerpt!r}") from error


def assert_port_vacant(port: int) -> None:
    """Fail before launch when the selected loopback port is already bound."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
        probe.listen(1)
    except OSError as error:
        raise BenchError(f"selected port 127.0.0.1:{port} is occupied: {error}") from error
    finally:
        probe.close()


def preflight_ports_vacant(ports: Iterable[int]) -> None:
    for port in ports:
        assert_port_vacant(port)


def wait_healthy(server: ServerHandle, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_error = "not yet queried"
    while time.monotonic() < deadline:
        exit_code = server.process.poll()
        if exit_code is not None:
            raise BenchError(
                f"server PID {server.process.pid} exited with {exit_code}; log: {server.log_path}"
            )
        try:
            # /health is a status endpoint; its exact JSON shape is irrelevant.
            http_json(url, timeout=2.0)
            exit_code = server.process.poll()
            if exit_code is not None:
                raise BenchError(
                    f"health responded after launched server PID {server.process.pid} "
                    f"exited with {exit_code}; refusing foreign/stale readiness"
                )
            return
        except BenchError as error:
            last_error = str(error)
            time.sleep(0.5)
    raise BenchError(
        f"server PID {server.process.pid} did not become healthy within {timeout:.1f}s: {last_error}; "
        f"log: {server.log_path}"
    )


def stop_server(server: ServerHandle) -> dict[str, Any]:
    """Stop exactly the process this boot started; never broad-match or pkill."""
    proc = server.process
    signal_used = None
    try:
        if proc.poll() is None:
            signal_used = "SIGTERM"
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                signal_used = "SIGKILL"
                proc.kill()
                proc.wait(timeout=30)
    finally:
        server.stream.flush()
        server.stream.close()
    return {"pid": proc.pid, "teardown_signal": signal_used, "exit_code": proc.returncode}


def log_tail(path: Path, max_bytes: int = 12_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(max(0, size - max_bytes))
        return source.read().decode("utf-8", "replace")


def get_metric_snapshot(port: int, raw_path: Path) -> dict[str, Any]:
    snapshot = http_json(f"http://127.0.0.1:{port}/metrics.json", timeout=15.0)
    if not isinstance(snapshot, dict):
        raise BenchError("/metrics.json returned a non-object JSON value")
    json_write(raw_path, snapshot)
    return snapshot


def number_at(root: dict[str, Any], *keys: str) -> float | None:
    value: Any = root
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def histogram_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Return sums/counts when the documented metrics JSON shape is present."""
    result: dict[str, Any] = {"available": True, "histograms": {}}
    for name in ("time_to_first_token_seconds", "decode_time_seconds"):
        before_sum = number_at(before, "histograms", name, "sum")
        after_sum = number_at(after, "histograms", name, "sum")
        before_count = number_at(before, "histograms", name, "count")
        after_count = number_at(after, "histograms", name, "count")
        if None in (before_sum, after_sum, before_count, after_count):
            result["available"] = False
            result["reason"] = f"missing numeric histograms.{name}.sum/count"
            return result
        result["histograms"][name] = {
            "sum_seconds": after_sum - before_sum,
            "count": int(after_count - before_count),
        }
    return result


def output_contract(response: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        choice = response["choices"][0]
        message = choice.get("message", {})
        content = message.get("content")
        if content is None:
            content = choice.get("text")
        if not isinstance(content, str):
            raise TypeError("choice has no text content")
    except (KeyError, IndexError, TypeError) as error:
        raise BenchError("completion response lacks choices[0] text content") from error

    usage = response.get("usage", {})
    timings = response.get("timings", {})
    contract = {
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens", timings.get("predicted_n")),
    }
    return contract, output_contract_sha256(contract)


def assess_engagement(
    arm: str,
    log: str,
    expected_dspark_stats: int,
    *,
    dspark_profile_required: bool = False,
) -> dict[str, Any]:
    spec_stats = log.count(DSPARK_STATS_MARKER)
    gpu_markers = log.count(GPU_MARKOV_MARKER)
    joined_markers = log.count(JOIN_VERIFY_MARKER)
    block_cap_b2_markers = log.count(BLOCK_CAP_B2_MARKER)
    replay_commit_markers = log.count(REPLAY_COMMIT_MARKER)
    profile_markers = log.count(DSPARK_PROFILE_MARKER)
    evidence = {
        "spec_stats_count": spec_stats,
        "gpu_markov_marker_count": gpu_markers,
        "joined_verify_marker_count": joined_markers,
        "block_cap_b2_marker_count": block_cap_b2_markers,
        "replay_commit_marker_count": replay_commit_markers,
        "dspark_profile_marker_count": profile_markers,
        "dspark_profile_required": dspark_profile_required,
        "expected_dspark_stats": expected_dspark_stats if arm != "S" else 0,
    }
    if arm == "S":
        if (
            spec_stats != 0
            or gpu_markers != 0
            or joined_markers != 0
            or replay_commit_markers != 0
        ):
            raise BenchError(f"S must have no speculative evidence, observed {evidence}")
    elif arm == "L":
        if (
            spec_stats != expected_dspark_stats
            or gpu_markers != 0
            or joined_markers != 0
            or replay_commit_markers != 0
        ):
            raise BenchError(f"L must have DSpark stats and no fast-path markers, observed {evidence}")
    elif arm == "G":
        if (
            spec_stats != expected_dspark_stats
            or gpu_markers != 1
            or joined_markers != 0
            or replay_commit_markers != 0
        ):
            raise BenchError(
                f"G must have DSpark stats, one GPU marker, and no joined marker, observed {evidence}"
            )
    elif arm == "J":
        if (
            spec_stats != expected_dspark_stats
            or gpu_markers != 1
            or joined_markers != 1
            or replay_commit_markers != 0
        ):
            raise BenchError(
                f"J must have DSpark stats and exactly one of both engagement markers, observed {evidence}"
            )
    elif arm == "B2":
        if (
            spec_stats != expected_dspark_stats
            or gpu_markers != 1
            or joined_markers != 0
            or block_cap_b2_markers != 1
            or replay_commit_markers != 0
        ):
            raise BenchError(
                "B2 must have DSpark stats, one GPU marker, one effective=2 cap marker, "
                f"and no joined marker, observed {evidence}"
            )
        if dspark_profile_required and profile_markers < 1:
            raise BenchError(f"B2 profiled run has no profile marker, observed {evidence}")
        if not dspark_profile_required and profile_markers != 0:
            raise BenchError(f"B2 unprofiled run leaked a profile marker, observed {evidence}")
    elif arm == "R2":
        if (
            spec_stats != expected_dspark_stats
            or gpu_markers != 1
            or joined_markers != 0
            or block_cap_b2_markers != 1
            or replay_commit_markers != 1
            or profile_markers != 0
        ):
            raise BenchError(
                "R2 must have DSpark stats, one GPU marker, one effective=2 cap marker, "
                "one replay-commit marker, no joined marker, and no profile marker; "
                f"observed {evidence}"
            )
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return evidence


def model_identity(model: Path, mode: str) -> dict[str, Any]:
    """Fingerprint the exact model without needlessly warming every weight file.

    ``metadata`` hashes each relative path, type, byte size, mtime_ns, and the
    content SHA-256 of small JSON/text configuration files.  ``full`` hashes
    every regular file, which is stronger but intentionally opt-in because it
    reads the entire (roughly 100 GB) model before the performance run.
    """
    entries: list[dict[str, Any]] = []
    config_hashes: dict[str, str] = {}
    for path in sorted(model.rglob("*")):
        relative = str(path.relative_to(model))
        stat = path.stat()
        entry: dict[str, Any] = {
            "path": relative,
            "kind": "dir" if path.is_dir() else "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if path.is_file() and (mode == "full" or path.suffix in {".json", ".txt"}):
            entry["sha256"] = sha256_file(path)
            if path.suffix in {".json", ".txt"}:
                config_hashes[relative] = entry["sha256"]
        entries.append(entry)
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(model),
        "hash_mode": mode,
        "tree_manifest_sha256": sha256_bytes(encoded),
        "config_file_sha256": config_hashes,
        "entry_count": len(entries),
    }


def measured_request(
    *,
    port: int,
    body: dict[str, Any],
    timeout: float,
    raw_dir: Path,
    request_id: str,
) -> dict[str, Any]:
    before_path = raw_dir / f"{request_id}.metrics-before.json"
    after_path = raw_dir / f"{request_id}.metrics-after.json"
    response_path = raw_dir / f"{request_id}.response.json"
    before = get_metric_snapshot(port, before_path)
    start_ns = time.monotonic_ns()
    response = http_json(
        f"http://127.0.0.1:{port}/v1/chat/completions", payload=body, timeout=timeout
    )
    wall_seconds = (time.monotonic_ns() - start_ns) / 1_000_000_000
    if not isinstance(response, dict):
        raise BenchError("completion endpoint returned a non-object JSON value")
    json_write(response_path, response)
    after = get_metric_snapshot(port, after_path)
    delta = histogram_delta(before, after)
    if not delta["available"]:
        raise BenchError(f"metrics histogram shape unavailable for {request_id}: {delta.get('reason')}")
    for name, values in delta["histograms"].items():
        if values["count"] != 1:
            raise BenchError(
                f"{request_id}: metrics {name} count delta must be 1, got {values['count']}"
            )
    contract, contract_hash = output_contract(response)
    timings = response.get("timings")
    usage = response.get("usage")
    return {
        "request_id": request_id,
        "wall_seconds": wall_seconds,
        "output_contract": contract,
        "output_contract_sha256": contract_hash,
        "usage": usage,
        "timings": timings,
        "metrics_histogram_delta": delta,
        "raw": {
            "response": str(response_path),
            "metrics_before": str(before_path),
            "metrics_after": str(after_path),
        },
    }


def warmup_request(
    *, port: int, body: dict[str, Any], timeout: float, raw_dir: Path, request_id: str
) -> None:
    response = http_json(
        f"http://127.0.0.1:{port}/v1/chat/completions", payload=body, timeout=timeout
    )
    if not isinstance(response, dict):
        raise BenchError("warmup completion endpoint returned a non-object JSON value")
    json_write(raw_dir / f"{request_id}.warmup-response.json", response)


def run_boot(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    arm: str,
    order: tuple[str, ...],
    selected_order: tuple[str, ...],
    order_index: int,
    order_position: int,
    selected_order_position: int,
    boot_index: int,
    port: int,
    measured_per_boot: int,
    prompt: str,
) -> dict[str, Any]:
    # Recheck immediately before Popen as well as in the all-port preflight.
    # This cannot eliminate an OS-level bind race, but it fails cleanly for a
    # stable occupant instead of accepting that process's readiness endpoint.
    assert_port_vacant(port)
    boot_id = f"boot-{boot_index:02d}-{arm}"
    boot_dir = out_dir / "boots" / boot_id
    raw_dir = boot_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    env, removed = sanitized_server_env(
        arm,
        args.execution_profile,
        args.dspark_profile_every,
        args.wired_mode,
    )
    argv = common_argv(args.binary, args.model, port, arm)
    body = request_body(args.model, prompt, arm, args.max_tokens)
    json_write(raw_dir / "request-body.json", body)
    json_write(
        boot_dir / "launch.json",
        {
            "boot_id": boot_id,
            "arm": arm,
            "arm_name": ARM_NAMES[arm],
            "execution_profile": args.execution_profile,
            "wired_policy": wired_policy_provenance(args.wired_mode),
            "balanced_order": "".join(order),
            "selected_order": "".join(selected_order),
            "order_index": order_index,
            "order_position": order_position,
            "selected_order_position": selected_order_position,
            "port": port,
            "argv": argv,
            "sanitized_inherited_mlx_serve_keys": removed,
            "server_mlx_serve_env": {key: env[key] for key in sorted(env) if key.startswith("MLX_SERVE_")},
            "warmups_excluded": args.warmups_per_boot,
            "measured_requests": measured_per_boot,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "timeouts_seconds": {
                "startup": args.startup_timeout,
                "request": args.request_timeout,
            },
            "effective_request_body": body,
        },
    )

    log_path = boot_dir / "server.log"
    stream = log_path.open("wb")
    server = ServerHandle(
        process=subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env),
        log_path=log_path,
        stream=stream,
    )
    result: dict[str, Any] = {
        "boot_id": boot_id,
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "balanced_order": "".join(order),
        "selected_order": "".join(selected_order),
        "order_index": order_index,
        "order_position": order_position,
        "selected_order_position": selected_order_position,
        "port": port,
        "pid": server.process.pid,
        "server_log": str(log_path),
        "measurements": [],
    }
    raised: BaseException | None = None
    try:
        wait_healthy(server, port, args.startup_timeout)
        for warmup_index in range(args.warmups_per_boot):
            warmup_request(
                port=port,
                body=body,
                timeout=args.request_timeout,
                raw_dir=raw_dir,
                request_id=f"warmup-{warmup_index:02d}",
            )
        for measurement_index in range(measured_per_boot):
            result["measurements"].append(
                measured_request(
                    port=port,
                    body=body,
                    timeout=args.request_timeout,
                    raw_dir=raw_dir,
                    request_id=f"measured-{measurement_index:02d}",
                )
            )
    except BaseException as error:
        raised = error
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["teardown"] = stop_server(server)
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        try:
            result["engagement"] = assess_engagement(
                arm,
                log,
                args.warmups_per_boot + len(result["measurements"]),
                dspark_profile_required=(
                    arm == "B2" and args.dspark_profile_every is not None
                ),
            )
        except BaseException as evidence_error:
            result["engagement_error"] = f"{type(evidence_error).__name__}: {evidence_error}"
            if raised is None:
                raised = evidence_error
        result["server_log_tail"] = log_tail(log_path)
        json_write(boot_dir / "result.json", result)
    if raised is not None:
        raise raised
    return result


def all_measurements(boots: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for boot in boots:
        for result in boot["measurements"]:
            measurements.append({"arm": boot["arm"], "boot_id": boot["boot_id"], **result})
    return measurements


def json_read(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchError(f"cannot read {description}: {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchError(f"invalid JSON in {description}: {path}: {error}") from error


def compatible_compare_receipt(receipt_dir: Path, current_manifest: dict[str, Any]) -> dict[str, Any]:
    """Load completed measurements from a previous, identity-compatible run.

    A failed prior run can still be a valid reference when its completed S
    boot was checkpointed in ``boots.json`` before the later arm failed.  The
    reference is deliberately used for output equality only; it is never mixed
    into this receipt's timing aggregates.
    """
    receipt_dir = receipt_dir.resolve()
    manifest_path = receipt_dir / "manifest.json"
    if not receipt_dir.is_dir() or not manifest_path.is_file():
        raise BenchError(
            "--compare-receipt must name a receipt directory containing manifest.json: "
            f"{receipt_dir}"
        )
    previous_manifest = json_read(manifest_path, "comparison receipt manifest")
    if not isinstance(previous_manifest, dict):
        raise BenchError(f"comparison receipt manifest is not an object: {manifest_path}")

    if (
        previous_manifest.get("schema") != MANIFEST_SCHEMA
        or previous_manifest.get("schema_version") != MANIFEST_VERSION
    ):
        raise BenchError(
            "comparison receipt uses an old/unknown manifest schema and is not "
            f"proof-grade (need {MANIFEST_SCHEMA} v{MANIFEST_VERSION}): {receipt_dir}"
        )

    checks = (
        ("prompt_sha256", previous_manifest.get("prompt_sha256"), current_manifest.get("prompt_sha256")),
        (
            "binary.sha256",
            number_or_string_at(previous_manifest, "binary", "sha256"),
            number_or_string_at(current_manifest, "binary", "sha256"),
        ),
        (
            "model.tree_manifest_sha256",
            number_or_string_at(previous_manifest, "model", "tree_manifest_sha256"),
            number_or_string_at(current_manifest, "model", "tree_manifest_sha256"),
        ),
        (
            "harness.sha256",
            number_or_string_at(previous_manifest, "harness", "sha256"),
            number_or_string_at(current_manifest, "harness", "sha256"),
        ),
        (
            "timeouts_seconds",
            previous_manifest.get("timeouts_seconds"),
            current_manifest.get("timeouts_seconds"),
        ),
        (
            "effective_request_config",
            previous_manifest.get("effective_request_config"),
            current_manifest.get("effective_request_config"),
        ),
        (
            "effective_arm_config",
            previous_manifest.get("effective_arm_config"),
            current_manifest.get("effective_arm_config"),
        ),
        (
            "execution_profile",
            previous_manifest.get("execution_profile"),
            current_manifest.get("execution_profile"),
        ),
        (
            "wired_policy",
            previous_manifest.get("wired_policy"),
            current_manifest.get("wired_policy"),
        ),
    )
    mismatches = [name for name, prior, current in checks if not prior or prior != current]
    if mismatches:
        raise BenchError(
            "comparison receipt is not identity-compatible (mismatch or missing "
            + ", ".join(mismatches)
            + "): "
            + str(receipt_dir)
        )
    for name, prior, current in checks[:4]:
        if not isinstance(prior, str) or LOWER_HEX_SHA256.fullmatch(prior) is None:
            raise BenchError(f"comparison receipt has invalid {name}: {receipt_dir}")
        if not isinstance(current, str) or LOWER_HEX_SHA256.fullmatch(current) is None:
            raise BenchError(f"current manifest has invalid {name}")

    measurements_path = receipt_dir / "measurements.json"
    boots_path = receipt_dir / "boots.json"
    if measurements_path.is_file():
        measurements = json_read(measurements_path, "comparison receipt measurements")
        measurement_source = str(measurements_path)
    elif boots_path.is_file():
        boots = json_read(boots_path, "comparison receipt completed boots")
        if not isinstance(boots, list):
            raise BenchError(f"comparison receipt boots.json is not a list: {boots_path}")
        measurements = all_measurements(boots)
        measurement_source = str(boots_path)
    else:
        raise BenchError(
            "comparison receipt has neither measurements.json nor completed boots.json: "
            f"{receipt_dir}"
        )
    if not isinstance(measurements, list) or not measurements:
        raise BenchError(f"comparison receipt contains no completed measurements: {receipt_dir}")

    validated: list[dict[str, Any]] = []
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, dict):
            raise BenchError(f"comparison measurement {index} is not an object")
        arm = measurement.get("arm")
        if arm not in ARM_NAMES:
            raise BenchError(f"comparison measurement {index} has invalid arm {arm!r}")
        if not isinstance(measurement.get("boot_id"), str) or not isinstance(
            measurement.get("request_id"), str
        ):
            raise BenchError(f"comparison measurement {index} lacks boot_id/request_id")
        verified_measurement_hash(measurement, f"comparison measurement {index}")
        validated.append(measurement)

    return {
        "receipt_dir": str(receipt_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "measurement_source": measurement_source,
        "measurement_count": len(validated),
        "arms": ordered_arms(measurement["arm"] for measurement in validated),
        "measurements": validated,
    }


def number_or_string_at(root: dict[str, Any], *keys: str) -> str | int | float | None:
    value: Any = root
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    return value


def assert_exact_output_equivalence(measurements: list[dict[str, Any]]) -> str:
    if not measurements:
        raise BenchError("no measured outputs; cannot establish equivalence")
    authenticated: list[tuple[dict[str, Any], str]] = []
    for index, item in enumerate(measurements):
        if not isinstance(item, dict):
            raise BenchError(f"measurement {index} is not an object")
        authenticated.append(
            (item, verified_measurement_hash(item, f"measurement {index}"))
        )
    hashes = {digest for _, digest in authenticated}
    if len(hashes) != 1:
        by_hash: dict[str, list[str]] = {}
        for item, digest in authenticated:
            by_hash.setdefault(digest, []).append(
                f"{item.get('boot_id', '?')}/{item.get('request_id', '?')}"
            )
        raise BenchError(
            "measured output contracts differ; speed summary intentionally withheld: "
            + json.dumps(by_hash, sort_keys=True)
        )
    return next(iter(hashes))


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def ordered_arms(arms: Iterable[str]) -> list[str]:
    arm_set = set(arms)
    return [arm for arm in ALL_ARMS if arm in arm_set]


def output_arm_coverage(
    measurements: list[dict[str, Any]],
    selected_arms: tuple[str, ...],
    compare_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    current = {item["arm"] for item in measurements}
    imported = set() if compare_receipt is None else set(compare_receipt["arms"])
    covered = current | imported
    # The explicit fast-default B2/R2 experiments have their own serial baselines.
    # Every other run retains S/L/G as the canonical baseline, with selected
    # extension arms additionally required for its requested result.
    selected_set = set(selected_arms)
    if selected_set == set(R2_ARMS):
        canonical_baseline = R2_ARMS
    elif selected_set == set(B2_ARMS):
        canonical_baseline = B2_ARMS
    else:
        canonical_baseline = DEFAULT_ARMS
    required = set(canonical_baseline) | set(selected_arms)
    missing = required - covered
    return {
        "canonical_baseline_arms": list(canonical_baseline),
        "selected_current_receipt_arms": list(selected_arms),
        "current_receipt_arms": ordered_arms(current),
        "comparison_receipt_arms": ordered_arms(imported),
        "covered_arms": ordered_arms(covered),
        "required_arms": ordered_arms(required),
        "missing_required_arms": ordered_arms(missing),
        "complete_for_required_arms": not missing,
    }


def schedule_assessment(
    mode: str,
    selected_arms: tuple[str, ...],
    expected_orders: tuple[tuple[str, ...], ...],
    observed_orders: tuple[tuple[str, ...], ...],
    measurements_complete: bool,
) -> dict[str, Any]:
    frozen_complete = observed_orders == expected_orders and measurements_complete
    if mode == "pilot":
        label = "one-pass pilot"
        counterbalanced = False
        position_balanced = False
    elif (
        set(selected_arms) == set(B2_ARMS)
        and expected_orders == B2_BALANCED_ORDERS
        and frozen_complete
    ):
        label = "complete two-order position-balanced S/B2 schedule"
        counterbalanced = False
        position_balanced = True
    elif (
        set(selected_arms) == set(R2_ARMS)
        and expected_orders == R2_BALANCED_ORDERS
        and frozen_complete
    ):
        label = "complete forward/reverse S/B2/R2 schedule"
        counterbalanced = False
        # The two orders invert S and R2 around the fixed B2 middle position.
        # This controls directionality without claiming full three-position balance.
        position_balanced = False
    elif (
        set(selected_arms) == set(DEFAULT_ARMS)
        and expected_orders == BALANCED_ORDERS
        and frozen_complete
    ):
        label = "complete frozen six-order counterbalanced S/L/G schedule"
        counterbalanced = True
        position_balanced = True
    elif (
        set(selected_arms) == set(LEGACY_ALL_ARMS)
        and expected_orders == FOUR_ARM_BALANCED_ORDERS
        and frozen_complete
    ):
        label = "complete frozen four-order position-balanced S/L/G/J schedule"
        counterbalanced = False
        position_balanced = True
    else:
        label = "balanced-mode recovery subset; not a complete canonical balanced schedule"
        counterbalanced = False
        position_balanced = False
    return {
        "mode": mode,
        "label": label,
        "expected_orders": ["".join(order) for order in expected_orders],
        "observed_orders": ["".join(order) for order in observed_orders],
        "complete_frozen_schedule": frozen_complete,
        "counterbalanced": counterbalanced,
        "position_balanced": position_balanced,
        "measurements_complete": measurements_complete,
    }


def observed_schedule_from_boots(boots: list[dict[str, Any]]) -> tuple[tuple[str, ...], ...]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for boot in boots:
        grouped.setdefault(int(boot["order_index"]), []).append(boot)
    observed: list[tuple[str, ...]] = []
    for order_index in sorted(grouped):
        rows = sorted(grouped[order_index], key=lambda row: int(row["selected_order_position"]))
        positions = [int(row["selected_order_position"]) for row in rows]
        if positions != list(range(len(rows))):
            raise BenchError(f"observed order {order_index} has non-contiguous positions {positions}")
        order = tuple(str(row["arm"]) for row in rows)
        declarations = {str(row["selected_order"]) for row in rows}
        if declarations != {"".join(order)}:
            raise BenchError(
                f"observed order {order_index} disagrees with boot declarations: {declarations} vs {order}"
            )
        observed.append(order)
    return tuple(observed)


def finite_ratio(numerator: Any, denominator: Any) -> float | None:
    """Return an intentionally directional metric ratio, or no result."""
    numerator_value = numeric(numerator)
    denominator_value = numeric(denominator)
    if numerator_value is None or denominator_value in (None, 0.0):
        return None
    return numerator_value / denominator_value


def arm_metric_ratio(
    arms: dict[str, dict[str, Any]], numerator_arm: str, denominator_arm: str
) -> dict[str, Any] | None:
    """Compare two named arms without constraining the receipt's other arms."""
    numerator = arms.get(numerator_arm)
    denominator = arms.get(denominator_arm)
    if numerator is None or denominator is None:
        return None
    metric_names = (
        "wall_tokens_per_second",
        "wall_seconds_per_completion_token",
        "metrics_decode_tokens_per_second",
        "metrics_decode_seconds_per_completion_token",
        "metrics_ttft_seconds_mean",
        "response_predicted_ms_mean",
    )
    return {
        "numerator_arm": numerator_arm,
        "denominator_arm": denominator_arm,
        "direction": "numerator / denominator",
        "metrics": {
            name: finite_ratio(numerator.get(name), denominator.get(name))
            for name in metric_names
        },
    }


def speed_summary(
    measurements: list[dict[str, Any]],
    output_hash: str,
    selected_arms: tuple[str, ...],
    compare_receipt: dict[str, Any] | None,
    *,
    mode: str,
    expected_orders: tuple[tuple[str, ...], ...],
    observed_orders: tuple[tuple[str, ...], ...],
    measured_per_boot: int,
    request_config: dict[str, Any],
    dspark_profile_every: int | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ALL_ARMS}
    for item in measurements:
        groups[item["arm"]].append(item)
    arms: dict[str, Any] = {}
    for arm, rows in groups.items():
        if not rows:
            continue
        wall_seconds = sum(float(row["wall_seconds"]) for row in rows)
        completion_tokens = sum(
            int(row["output_contract"].get("completion_tokens") or 0) for row in rows
        )
        ttft_seconds = sum(
            float(row["metrics_histogram_delta"]["histograms"]["time_to_first_token_seconds"]["sum_seconds"])
            for row in rows
        )
        decode_seconds = sum(
            float(row["metrics_histogram_delta"]["histograms"]["decode_time_seconds"]["sum_seconds"])
            for row in rows
        )
        response_decode_ms = []
        for row in rows:
            timings = row.get("timings")
            if isinstance(timings, dict):
                value = numeric(timings.get("predicted_ms"))
                if value is not None:
                    response_decode_ms.append(value)
        arms[arm] = {
            "name": ARM_NAMES[arm],
            "samples": len(rows),
            "completion_tokens": completion_tokens,
            "wall_seconds_total": wall_seconds,
            "wall_tokens_per_second": (completion_tokens / wall_seconds) if wall_seconds else None,
            "wall_seconds_per_completion_token": (
                wall_seconds / completion_tokens if completion_tokens else None
            ),
            "metrics_ttft_seconds_total": ttft_seconds,
            "metrics_ttft_seconds_mean": (ttft_seconds / len(rows)) if rows else None,
            "metrics_decode_seconds_total": decode_seconds,
            "metrics_decode_tokens_per_second": (
                completion_tokens / decode_seconds if decode_seconds else None
            ),
            "metrics_decode_seconds_per_completion_token": (
                decode_seconds / completion_tokens if completion_tokens else None
            ),
            "response_predicted_ms_mean": (
                sum(response_decode_ms) / len(response_decode_ms) if response_decode_ms else None
            ),
        }
    expected_counts = {
        arm: sum(order.count(arm) for order in expected_orders) * measured_per_boot
        for arm in ALL_ARMS
    }
    observed_counts = {arm: len(groups[arm]) for arm in ALL_ARMS}
    measurements_complete = all(observed_counts[arm] == expected_counts[arm] for arm in ALL_ARMS)
    schedule = schedule_assessment(
        mode, selected_arms, expected_orders, observed_orders, measurements_complete
    )
    coverage = output_arm_coverage(measurements, selected_arms, compare_receipt)
    if coverage["complete_for_required_arms"]:
        cross_arm = "exact match across all required arms: " + ",".join(coverage["required_arms"])
    else:
        cross_arm = (
            "exact match only across covered arms "
            + ",".join(coverage["covered_arms"])
            + "; missing required arms "
            + ",".join(coverage["missing_required_arms"])
        )
    if mode == "pilot":
        speed_scope = "same-receipt one-pass pilot; directional timing only"
    elif schedule["counterbalanced"]:
        speed_scope = "same-receipt complete counterbalanced S/L/G timing comparison"
    elif (
        set(selected_arms) == set(B2_ARMS)
        and expected_orders == B2_BALANCED_ORDERS
        and schedule["position_balanced"]
    ):
        speed_scope = "same-receipt complete position-balanced S/B2 timing comparison"
    elif (
        set(selected_arms) == set(R2_ARMS)
        and expected_orders == R2_BALANCED_ORDERS
        and schedule["complete_frozen_schedule"]
    ):
        speed_scope = "same-receipt complete forward/reverse S/B2/R2 timing comparison"
    elif schedule["position_balanced"]:
        speed_scope = "same-receipt complete position-balanced S/L/G/J timing comparison"
    else:
        speed_scope = "current-receipt recovery/subset timing only; not a canonical balanced comparison"
    if compare_receipt is not None:
        speed_scope += "; imported receipt is used for output equality only, never timing"
    if dspark_profile_every is not None:
        speed_scope = (
            f"B2 phase profiling every {dspark_profile_every} round(s); "
            "timing is diagnostic, not a symmetric speed comparison; "
            + speed_scope
        )
    return {
        "output_contract_sha256": output_hash,
        "output_arm_coverage": coverage,
        "schedule": schedule,
        "comparability": {
            "same_binary": True,
            "same_model_directory": True,
            "same_prompt": True,
            "effective_request_config": request_config,
            "speed_reported_only_after_exact_output_contract_match": True,
            "selected_arms": list(selected_arms),
            "b2_dspark_profile_every": dspark_profile_every,
            "cross_arm_output_equivalence": cross_arm,
            "cross_arm_speed_scope": speed_scope,
            "comparison_receipt": None
            if compare_receipt is None
            else {
                "receipt_dir": compare_receipt["receipt_dir"],
                "manifest_sha256": compare_receipt["manifest_sha256"],
                "measurement_source": compare_receipt["measurement_source"],
                "measurement_count": compare_receipt["measurement_count"],
                "arms": compare_receipt["arms"],
                "used_for_output_gate_only": True,
            },
        },
        "arms": arms,
        "arm_metric_ratios": {
            "B2_vs_R2": arm_metric_ratio(arms, "B2", "R2"),
            "S_vs_R2": arm_metric_ratio(arms, "S", "R2"),
        },
    }


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is None:
        return args.prompt
    if args.prompt != DEFAULT_PROMPT:
        raise BenchError("use either --prompt or --prompt-file, not both")
    return args.prompt_file.read_text(encoding="utf-8")


def validate_experiment_contract(
    execution_profile: str,
    selected_arms: tuple[str, ...],
    mode: str,
    dspark_profile_every: int | None,
) -> None:
    selected = set(selected_arms)
    is_b2_pair = selected == set(B2_ARMS)
    is_r2_triplet = selected == set(R2_ARMS)
    if execution_profile == "fast-default" and not (is_b2_pair or is_r2_triplet):
        raise BenchError(
            "--execution-profile fast-default is reserved for the explicit "
            "--arms S,B2 or --arms S,B2,R2 experiments"
        )
    if "R2" in selected and not is_r2_triplet:
        raise BenchError("R2 requires the explicit replay comparison --arms S,B2,R2")
    if "B2" in selected and not (is_b2_pair or is_r2_triplet):
        raise BenchError("B2 requires --arms S,B2 or --arms S,B2,R2")
    if ("B2" in selected or "R2" in selected) and execution_profile != "fast-default":
        raise BenchError("B2 and R2 require --execution-profile fast-default")
    if dspark_profile_every is not None:
        if not is_b2_pair:
            raise BenchError("--dspark-profile-every is available only for the S,B2 experiment")
        if not 1 <= dspark_profile_every <= 1024:
            raise BenchError("--dspark-profile-every must be in 1..1024")
        if mode != "pilot":
            raise BenchError(
                "--dspark-profile-every is pilot-only; balanced speed receipts must be unprofiled"
            )


def validate_args(
    args: argparse.Namespace,
    selected_arms: tuple[str, ...],
    selected_schedule: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
) -> None:
    validate_experiment_contract(
        args.execution_profile,
        selected_arms,
        args.mode,
        args.dspark_profile_every,
    )
    if args.warmups_per_boot != 1:
        raise BenchError(
            "--warmups-per-boot must be exactly 1: every fresh boot has one excluded same-shape warmup"
        )
    if args.measured_per_boot is not None and args.measured_per_boot != 1:
        raise BenchError("--measured-per-boot must be exactly 1 per fresh boot")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise BenchError("timeouts must be positive")
    if args.max_tokens < 1:
        raise BenchError("--max-tokens must be at least 1")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise BenchError(f"ReleaseFast binary is not executable: {args.binary}")
    if not args.model.is_dir():
        raise BenchError(f"model directory does not exist: {args.model}")
    if args.compare_receipt is not None and not args.compare_receipt.is_dir():
        raise BenchError(f"--compare-receipt directory does not exist: {args.compare_receipt}")
    boots = sum(len(effective_order) for _, effective_order in selected_schedule)
    port_base = args.port_base
    if port_base is None:
        port_base = 18_000 + (os.getpid() % 30_000)
        args.port_base = port_base
    if port_base < 1024 or port_base + boots > 65_535:
        raise BenchError(f"--port-base {port_base} cannot provide {boots} valid unique ports")
    preflight_ports_vacant(range(port_base, port_base + boots))


def prepare_out_dir(args: argparse.Namespace) -> Path:
    if args.out is None:
        args.out = Path("receipts/dsv4") / f"dspark-fast-path-{now_label()}-pid{os.getpid()}"
    out_dir = args.out
    if out_dir.exists() and any(out_dir.iterdir()):
        raise BenchError(f"refusing to overwrite non-empty receipt directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_benchmark(args: argparse.Namespace) -> Path:
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    if args.compare_receipt is not None:
        args.compare_receipt = args.compare_receipt.resolve()
    selected_arms = parse_selected_arms(args.arms)
    schedule = schedule_for(args.mode, selected_arms)
    selected_schedule = filtered_schedule(schedule, selected_arms)
    validate_args(args, selected_arms, selected_schedule)
    prompt = resolve_prompt(args)
    if not prompt.strip():
        raise BenchError("fixed benchmark prompt is empty")
    measured_per_boot = 1 if args.measured_per_boot is None else args.measured_per_boot

    out_dir = prepare_out_dir(args)
    request_config = effective_request_config(args.model, args.max_tokens)
    run_manifest = {
        **manifest_provenance(
            Path(__file__), args.startup_timeout, args.request_timeout, request_config
        ),
        "started_local": now_label(),
        "mode": args.mode,
        "orders": ["".join(order) for order in schedule],
        "selected_arms": list(selected_arms),
        "effective_orders": ["".join(effective) for _, effective in selected_schedule],
        "warmups_excluded_per_boot": args.warmups_per_boot,
        "measured_requests_per_boot": measured_per_boot,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "prompt_characters": len(prompt),
        "binary": {"path": str(args.binary), "sha256": sha256_file(args.binary)},
        "model": model_identity(args.model, args.model_hash_mode),
        "common_server_argv": common_argv(args.binary, args.model, args.port_base, "S")[1:],
        "execution_profile": profile_provenance(args.execution_profile),
        "wired_policy": wired_policy_provenance(args.wired_mode),
        "effective_arm_config": effective_arm_config(args.dspark_profile_every),
    }
    compare_receipt = (
        compatible_compare_receipt(args.compare_receipt, run_manifest)
        if args.compare_receipt is not None
        else None
    )
    if compare_receipt is not None:
        # Preserve enough source provenance to audit the imported equality
        # evidence without copying its large raw response/log tree.
        json_write(
            out_dir / "comparison-receipt.json",
            {
                key: value
                for key, value in compare_receipt.items()
                if key != "measurements"
            },
        )
        run_manifest["comparison_receipt"] = {
            key: value for key, value in compare_receipt.items() if key != "measurements"
        }
    json_write(out_dir / "manifest.json", run_manifest)

    boots: list[dict[str, Any]] = []
    try:
        boot_index = 0
        for order_index, (order, effective_order) in enumerate(selected_schedule):
            for selected_order_position, arm in enumerate(effective_order):
                order_position = order.index(arm)
                boot = run_boot(
                    args=args,
                    out_dir=out_dir,
                    arm=arm,
                    order=order,
                    selected_order=effective_order,
                    order_index=order_index,
                    order_position=order_position,
                    selected_order_position=selected_order_position,
                    boot_index=boot_index,
                    port=args.port_base + boot_index,
                    measured_per_boot=measured_per_boot,
                    prompt=prompt,
                )
                boots.append(boot)
                json_write(out_dir / "boots.json", boots)
                boot_index += 1
    except BaseException as error:
        json_write(
            out_dir / "failure.json",
            {"error": f"{type(error).__name__}: {error}", "completed_boots": len(boots)},
        )
        raise

    measurements = all_measurements(boots)
    json_write(out_dir / "measurements.json", measurements)
    output_gate_measurements = measurements + (
        [] if compare_receipt is None else compare_receipt["measurements"]
    )
    output_hash = assert_exact_output_equivalence(output_gate_measurements)
    expected_orders = tuple(effective for _, effective in selected_schedule)
    observed_orders = observed_schedule_from_boots(boots)
    summary = speed_summary(
        measurements,
        output_hash,
        selected_arms,
        compare_receipt,
        mode=args.mode,
        expected_orders=expected_orders,
        observed_orders=observed_orders,
        measured_per_boot=measured_per_boot,
        request_config=request_config,
        dspark_profile_every=args.dspark_profile_every,
    )
    summary["receipt_dir"] = str(out_dir)
    json_write(out_dir / "summary.json", summary)
    return out_dir


def run_self_tests() -> None:
    def must_fail(action: Any, label: str) -> None:
        try:
            action()
        except BenchError:
            return
        raise AssertionError(f"{label} must fail")

    default_args = parse_args([])
    assert default_args.wired_mode == "default"
    assert parse_args(["--wired-mode", "off"]).wired_mode == "off"
    help_output = io.StringIO()
    try:
        with contextlib.redirect_stdout(help_output):
            parse_args(["--help"])
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError("--help must exit successfully")
    assert "--wired-mode {default,off}" in help_output.getvalue()
    invalid_output = io.StringIO()
    try:
        with contextlib.redirect_stderr(invalid_output):
            parse_args(["--wired-mode", "invalid"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("invalid --wired-mode must fail argument validation")
    assert "invalid choice" in invalid_output.getvalue()

    assert schedule_for("pilot") == (ALL_ARMS,)
    assert schedule_for("balanced") == BALANCED_ORDERS
    assert schedule_for("balanced", LEGACY_ALL_ARMS) == FOUR_ARM_BALANCED_ORDERS
    assert schedule_for("balanced", B2_ARMS) == B2_BALANCED_ORDERS
    assert schedule_for("balanced", R2_ARMS) == R2_BALANCED_ORDERS
    assert [arm for order in BALANCED_ORDERS for arm in order].count("S") == 6
    assert [arm for order in BALANCED_ORDERS for arm in order].count("L") == 6
    assert [arm for order in BALANCED_ORDERS for arm in order].count("G") == 6
    for arm in LEGACY_ALL_ARMS:
        positions = [order.index(arm) for order in FOUR_ARM_BALANCED_ORDERS]
        assert sorted(positions) == [0, 1, 2, 3]
    for arm in B2_ARMS:
        positions = [order.index(arm) for order in B2_BALANCED_ORDERS]
        assert sorted(positions) == [0, 1]
    assert R2_BALANCED_ORDERS == (("S", "B2", "R2"), ("R2", "B2", "S"))
    assert parse_selected_arms(None) == DEFAULT_ARMS
    assert parse_selected_arms(["G"]) == ("G",)
    assert parse_selected_arms(["S,J"]) == ("S", "J")
    assert parse_selected_arms(["S,L", "G"]) == DEFAULT_ARMS
    assert parse_selected_arms(["S,B2"]) == B2_ARMS
    assert parse_selected_arms(["S,B2,R2"]) == R2_ARMS
    assert filtered_schedule(schedule_for("pilot"), ("G",)) == ((ALL_ARMS, ("G",)),)
    assert filtered_schedule(schedule_for("pilot", ("S", "J")), ("S", "J")) == (
        (ALL_ARMS, ("S", "J")),
    )
    balanced_g = filtered_schedule(schedule_for("balanced"), ("G",))
    assert len(balanced_g) == 6 and all(effective == ("G",) for _, effective in balanced_g)
    assert filtered_schedule(schedule_for("balanced", B2_ARMS), B2_ARMS) == (
        (("S", "B2"), ("S", "B2")),
        (("B2", "S"), ("B2", "S")),
    )
    assert filtered_schedule(schedule_for("balanced", R2_ARMS), R2_ARMS) == (
        (("S", "B2", "R2"), ("S", "B2", "R2")),
        (("R2", "B2", "S"), ("R2", "B2", "S")),
    )

    pilot_label = schedule_assessment(
        "pilot", ("S", "J"), (("S", "J"),), (("S", "J"),), True
    )
    assert pilot_label["label"] == "one-pass pilot"
    assert not pilot_label["counterbalanced"]
    balanced_label = schedule_assessment(
        "balanced", DEFAULT_ARMS, BALANCED_ORDERS, BALANCED_ORDERS, True
    )
    assert balanced_label["counterbalanced"]
    incomplete_label = schedule_assessment(
        "balanced", DEFAULT_ARMS, BALANCED_ORDERS, BALANCED_ORDERS[:-1], True
    )
    assert not incomplete_label["counterbalanced"]
    four_label = schedule_assessment(
        "balanced", LEGACY_ALL_ARMS, FOUR_ARM_BALANCED_ORDERS, FOUR_ARM_BALANCED_ORDERS, True
    )
    assert four_label["position_balanced"] and not four_label["counterbalanced"]
    b2_label = schedule_assessment(
        "balanced", B2_ARMS, B2_BALANCED_ORDERS, B2_BALANCED_ORDERS, True
    )
    assert b2_label["position_balanced"] and not b2_label["counterbalanced"]
    r2_label = schedule_assessment(
        "balanced", R2_ARMS, R2_BALANCED_ORDERS, R2_BALANCED_ORDERS, True
    )
    assert r2_label["label"] == "complete forward/reverse S/B2/R2 schedule"
    assert not r2_label["position_balanced"] and not r2_label["counterbalanced"]
    pilot_body = request_body(Path("/models/fake"), "prompt", "J", 16)
    assert pilot_body["max_tokens"] == 16 and pilot_body["enable_mtp"] is True

    inherited_keys = (
        "MLX_SERVE_DSPARK_SERIAL_VERIFY",
        BLOCK_CAP_ENV,
        DSPARK_PROFILE_ENV,
        DSPARK_PROFILE_EVERY_ENV,
        REPLAY_COMMIT_ENV,
        WIRED_ENV,
        "MLX_SERVE_DSV4_DEC_CHAIN",
    )
    previous = {key: os.environ.get(key) for key in inherited_keys}
    os.environ["MLX_SERVE_DSPARK_SERIAL_VERIFY"] = "1"
    os.environ[BLOCK_CAP_ENV] = "5"
    os.environ[DSPARK_PROFILE_ENV] = "1"
    os.environ[DSPARK_PROFILE_EVERY_ENV] = "99"
    os.environ[REPLAY_COMMIT_ENV] = "0"
    os.environ[WIRED_ENV] = "on"
    os.environ["MLX_SERVE_DSV4_DEC_CHAIN"] = "0"
    try:
        legacy, removed = sanitized_server_env("L")
        gpu, _ = sanitized_server_env("G")
        joined, _ = sanitized_server_env("J")
        fast_serial, fast_removed = sanitized_server_env("S", "fast-default")
        b2, _ = sanitized_server_env("B2", "fast-default")
        profiled_b2, _ = sanitized_server_env("B2", "fast-default", 1)
        r2, _ = sanitized_server_env("R2", "fast-default")
        assert "MLX_SERVE_DSPARK_SERIAL_VERIFY" in removed
        assert BLOCK_CAP_ENV in fast_removed
        assert DSPARK_PROFILE_ENV in fast_removed
        assert REPLAY_COMMIT_ENV in fast_removed
        assert WIRED_ENV in fast_removed
        assert "MLX_SERVE_DSPARK_SERIAL_VERIFY" not in legacy
        assert {key: legacy[key] for key in COMMON_ENV} == COMMON_ENV
        assert all(key in COMMON_ENV for key in legacy if key.startswith("MLX_SERVE_"))
        assert gpu[GPU_MARKOV_ENV] == "1"
        assert JOIN_VERIFY_ENV not in gpu
        assert joined[GPU_MARKOV_ENV] == "1"
        assert joined[JOIN_VERIFY_ENV] == "1"
        assert not any(key.startswith("MLX_SERVE_") for key in fast_serial)
        assert {
            key: value for key, value in b2.items() if key.startswith("MLX_SERVE_")
        } == {
            GPU_MARKOV_ENV: "1",
            BLOCK_CAP_ENV: "2",
        }
        assert {
            key: value for key, value in profiled_b2.items() if key.startswith("MLX_SERVE_")
        } == {
            GPU_MARKOV_ENV: "1",
            BLOCK_CAP_ENV: "2",
            DSPARK_PROFILE_ENV: "1",
            DSPARK_PROFILE_EVERY_ENV: "1",
        }
        assert {
            key: value for key, value in r2.items() if key.startswith("MLX_SERVE_")
        } == {
            GPU_MARKOV_ENV: "1",
            BLOCK_CAP_ENV: "2",
            REPLAY_COMMIT_ENV: "1",
        }
        for arm in ALL_ARMS:
            profile = "fast-default" if arm in ("B2", "R2") else "conservative"
            default_wired, default_removed = sanitized_server_env(
                arm, profile, wired_mode="default"
            )
            off_wired, off_removed = sanitized_server_env(
                arm, profile, wired_mode="off"
            )
            assert WIRED_ENV in default_removed and WIRED_ENV in off_removed
            assert WIRED_ENV not in default_wired
            assert off_wired[WIRED_ENV] == "off"
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert profile_provenance("conservative")["common_mlx_serve_env"] == COMMON_ENV
    assert profile_provenance("fast-default")["common_mlx_serve_env"] == {}
    assert wired_policy_provenance("default")["common_mlx_serve_env"] == {}
    assert wired_policy_provenance("off")["common_mlx_serve_env"] == {
        WIRED_ENV: "off"
    }
    assert effective_arm_config()["B2"]["extra_mlx_serve_env"] == {
        GPU_MARKOV_ENV: "1",
        BLOCK_CAP_ENV: "2",
    }
    assert effective_arm_config(8)["B2"]["extra_mlx_serve_env"][DSPARK_PROFILE_EVERY_ENV] == "8"
    assert effective_arm_config()["R2"]["extra_mlx_serve_env"] == {
        GPU_MARKOV_ENV: "1",
        BLOCK_CAP_ENV: "2",
        REPLAY_COMMIT_ENV: "1",
    }

    validate_experiment_contract("conservative", DEFAULT_ARMS, "balanced", None)
    validate_experiment_contract("fast-default", B2_ARMS, "pilot", None)
    validate_experiment_contract("fast-default", B2_ARMS, "pilot", 1)
    validate_experiment_contract("fast-default", R2_ARMS, "balanced", None)
    must_fail(
        lambda: validate_experiment_contract("conservative", B2_ARMS, "pilot", None),
        "B2 conservative profile",
    )
    must_fail(
        lambda: validate_experiment_contract("conservative", R2_ARMS, "balanced", None),
        "R2 conservative profile",
    )
    must_fail(
        lambda: validate_experiment_contract("fast-default", ("S", "R2"), "balanced", None),
        "R2 missing B2 control",
    )
    must_fail(
        lambda: validate_experiment_contract("fast-default", DEFAULT_ARMS, "pilot", None),
        "fast-default legacy arms",
    )
    must_fail(
        lambda: validate_experiment_contract("fast-default", ("S", "B2"), "balanced", 1),
        "profiled B2 balanced receipt",
    )
    must_fail(
        lambda: validate_experiment_contract("fast-default", B2_ARMS, "pilot", 0),
        "zero B2 profile cadence",
    )

    before = {
        "histograms": {
            "time_to_first_token_seconds": {"sum": 2.0, "count": 3},
            "decode_time_seconds": {"sum": 5.0, "count": 3},
        }
    }
    after = {
        "histograms": {
            "time_to_first_token_seconds": {"sum": 2.25, "count": 4},
            "decode_time_seconds": {"sum": 6.0, "count": 4},
        }
    }
    delta = histogram_delta(before, after)
    assert delta["available"]
    assert delta["histograms"]["time_to_first_token_seconds"]["count"] == 1
    assert delta["histograms"]["decode_time_seconds"]["sum_seconds"] == 1.0

    assess_engagement("S", "normal serial log", 0)
    assess_engagement("L", f"{DSPARK_STATS_MARKER}\n{DSPARK_STATS_MARKER}", 2)
    assess_engagement("G", f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}", 1)
    assess_engagement(
        "J", f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{JOIN_VERIFY_MARKER}", 1
    )
    assess_engagement(
        "B2",
        f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{BLOCK_CAP_B2_MARKER}",
        1,
    )
    assess_engagement(
        "B2",
        f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{BLOCK_CAP_B2_MARKER}\n{DSPARK_PROFILE_MARKER}",
        1,
        dspark_profile_required=True,
    )
    assess_engagement(
        "R2",
        f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{BLOCK_CAP_B2_MARKER}\n{REPLAY_COMMIT_MARKER}",
        1,
    )
    must_fail(lambda: assess_engagement("G", DSPARK_STATS_MARKER, 1), "G marker omission")
    must_fail(
        lambda: assess_engagement(
            "J", f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}", 1
        ),
        "J joined-marker omission",
    )
    must_fail(
        lambda: assess_engagement("B2", f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}", 1),
        "B2 cap-marker omission",
    )
    must_fail(
        lambda: assess_engagement(
            "B2",
            f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{BLOCK_CAP_B2_MARKER}",
            1,
            dspark_profile_required=True,
        ),
        "B2 profile-marker omission",
    )
    must_fail(
        lambda: assess_engagement(
            "R2", f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{BLOCK_CAP_B2_MARKER}", 1
        ),
        "R2 replay-marker omission",
    )

    serial_response = {
        "choices": [{"message": {"content": "exact"}, "finish_reason": "length"}],
        "usage": {"completion_tokens": 64},
    }
    contract, digest = output_contract(serial_response)
    assert contract["content"] == "exact"
    assert LOWER_HEX_SHA256.fullmatch(digest)

    def measurement(arm: str, *, claimed: str = digest) -> dict[str, Any]:
        return {
            "arm": arm,
            "boot_id": f"boot-00-{arm}",
            "request_id": "measured-00",
            "output_contract": contract,
            "output_contract_sha256": claimed,
        }

    assert_exact_output_equivalence([measurement("S"), measurement("G")])
    assert (
        assert_exact_output_equivalence(
            [measurement("S"), measurement("L"), measurement("G")]
        )
        == digest
    )
    must_fail(
        lambda: assert_exact_output_equivalence([measurement("S", claimed="A" * 64)]),
        "uppercase digest",
    )
    must_fail(
        lambda: assert_exact_output_equivalence([measurement("S", claimed="0" * 64)]),
        "tampered digest",
    )

    missing_l = output_arm_coverage(
        [measurement("G")], ("G",), {"arms": ["S"]}
    )
    assert missing_l["covered_arms"] == ["S", "G"]
    assert missing_l["missing_required_arms"] == ["L"]
    full_match = output_arm_coverage(
        [measurement("G")], ("G",), {"arms": ["S", "L"]}
    )
    assert full_match["covered_arms"] == ["S", "L", "G"]
    assert full_match["missing_required_arms"] == []
    assert full_match["complete_for_required_arms"]
    b2_match = output_arm_coverage(
        [measurement("B2")], B2_ARMS, {"arms": ["S"]}
    )
    assert b2_match["canonical_baseline_arms"] == ["S", "B2"]
    assert b2_match["covered_arms"] == ["S", "B2"]
    assert b2_match["missing_required_arms"] == []
    assert b2_match["complete_for_required_arms"]
    r2_match = output_arm_coverage(
        [measurement("B2"), measurement("R2")], R2_ARMS, {"arms": ["S"]}
    )
    assert r2_match["canonical_baseline_arms"] == ["S", "B2", "R2"]
    assert r2_match["covered_arms"] == ["S", "B2", "R2"]
    assert r2_match["complete_for_required_arms"]

    def speed_measurement(arm: str) -> dict[str, Any]:
        return {
            **measurement(arm),
            "wall_seconds": 1.0,
            "metrics_histogram_delta": {
                "histograms": {
                    "time_to_first_token_seconds": {"sum_seconds": 0.1},
                    "decode_time_seconds": {"sum_seconds": 0.9},
                }
            },
            "timings": {"predicted_ms": 900.0},
        }

    def comparison_stub(arms: list[str]) -> dict[str, Any]:
        return {
            "arms": arms,
            "receipt_dir": "/prior",
            "manifest_sha256": "d" * 64,
            "measurement_source": "/prior/measurements.json",
            "measurement_count": len(arms),
        }

    request_config = effective_request_config(Path("/models/fake"), 64)
    partial_summary = speed_summary(
        [speed_measurement("G")],
        digest,
        ("G",),
        comparison_stub(["S"]),
        mode="pilot",
        expected_orders=(("G",),),
        observed_orders=(("G",),),
        measured_per_boot=1,
        request_config=request_config,
    )
    assert partial_summary["schedule"]["label"] == "one-pass pilot"
    assert partial_summary["output_arm_coverage"]["covered_arms"] == ["S", "G"]
    assert partial_summary["output_arm_coverage"]["missing_required_arms"] == ["L"]
    complete_summary = speed_summary(
        [speed_measurement("G")],
        digest,
        ("G",),
        comparison_stub(["S", "L"]),
        mode="pilot",
        expected_orders=(("G",),),
        observed_orders=(("G",),),
        measured_per_boot=1,
        request_config=request_config,
    )
    assert complete_summary["output_arm_coverage"]["complete_for_required_arms"]
    b2_summary = speed_summary(
        [
            speed_measurement("S"),
            speed_measurement("B2"),
            speed_measurement("B2"),
            speed_measurement("S"),
        ],
        digest,
        B2_ARMS,
        None,
        mode="balanced",
        expected_orders=B2_BALANCED_ORDERS,
        observed_orders=B2_BALANCED_ORDERS,
        measured_per_boot=1,
        request_config=request_config,
    )
    assert b2_summary["schedule"]["position_balanced"]
    assert b2_summary["output_arm_coverage"]["complete_for_required_arms"]
    assert b2_summary["comparability"]["cross_arm_speed_scope"] == (
        "same-receipt complete position-balanced S/B2 timing comparison"
    )
    r2_summary = speed_summary(
        [
            speed_measurement("S"),
            speed_measurement("B2"),
            speed_measurement("R2"),
            speed_measurement("R2"),
            speed_measurement("B2"),
            speed_measurement("S"),
        ],
        digest,
        R2_ARMS,
        None,
        mode="balanced",
        expected_orders=R2_BALANCED_ORDERS,
        observed_orders=R2_BALANCED_ORDERS,
        measured_per_boot=1,
        request_config=request_config,
    )
    assert set(r2_summary["arms"]) == {"S", "B2", "R2"}
    assert r2_summary["schedule"]["label"] == "complete forward/reverse S/B2/R2 schedule"
    assert r2_summary["comparability"]["cross_arm_speed_scope"] == (
        "same-receipt complete forward/reverse S/B2/R2 timing comparison"
    )
    assert r2_summary["arm_metric_ratios"]["B2_vs_R2"] == {
        "numerator_arm": "B2",
        "denominator_arm": "R2",
        "direction": "numerator / denominator",
        "metrics": {
            "wall_tokens_per_second": 1.0,
            "wall_seconds_per_completion_token": 1.0,
            "metrics_decode_tokens_per_second": 1.0,
            "metrics_decode_seconds_per_completion_token": 1.0,
            "metrics_ttft_seconds_mean": 1.0,
            "response_predicted_ms_mean": 1.0,
        },
    }
    assert r2_summary["arm_metric_ratios"]["S_vs_R2"]["metrics"][
        "wall_tokens_per_second"
    ] == 1.0

    provenance = manifest_provenance(Path(__file__), 900.0, 901.0, request_config)
    assert provenance["schema"] == MANIFEST_SCHEMA
    assert provenance["schema_version"] == MANIFEST_VERSION
    assert LOWER_HEX_SHA256.fullmatch(provenance["harness"]["sha256"])
    assert provenance["timeouts_seconds"] == {"startup": 900.0, "request": 901.0}
    assert provenance["effective_request_config"]["max_tokens"] == 64
    current_manifest = {
        **provenance,
        "prompt_sha256": "a" * 64,
        "binary": {"sha256": "b" * 64},
        "model": {"tree_manifest_sha256": "c" * 64},
        "effective_arm_config": effective_arm_config(),
        "execution_profile": profile_provenance("conservative"),
        "wired_policy": wired_policy_provenance("default"),
    }
    with tempfile.TemporaryDirectory() as temp:
        receipt = Path(temp)
        json_write(receipt / "manifest.json", current_manifest)
        json_write(receipt / "measurements.json", [measurement("S")])
        imported = compatible_compare_receipt(receipt, current_manifest)
        assert imported["arms"] == ["S"]
        assert imported["measurement_count"] == 1
        assert_exact_output_equivalence(
            [measurement("G")] + imported["measurements"]
        )

        json_write(receipt / "measurements.json", [measurement("S", claimed="0" * 64)])
        must_fail(
            lambda: compatible_compare_receipt(receipt, current_manifest),
            "tampered imported hash",
        )
        hash_only = measurement("S")
        hash_only.pop("output_contract")
        json_write(receipt / "measurements.json", [hash_only])
        must_fail(
            lambda: compatible_compare_receipt(receipt, current_manifest),
            "legacy hash-only imported measurement",
        )
        json_write(receipt / "measurements.json", [measurement("S")])
        mismatched_manifest = json.loads(json.dumps(current_manifest))
        mismatched_manifest["effective_request_config"]["max_tokens"] = 16
        must_fail(
            lambda: compatible_compare_receipt(receipt, mismatched_manifest),
            "comparison max_tokens mismatch",
        )
        mismatched_manifest = json.loads(json.dumps(current_manifest))
        mismatched_manifest["execution_profile"] = profile_provenance("fast-default")
        must_fail(
            lambda: compatible_compare_receipt(receipt, mismatched_manifest),
            "comparison execution profile mismatch",
        )
        mismatched_manifest = json.loads(json.dumps(current_manifest))
        mismatched_manifest["wired_policy"] = wired_policy_provenance("off")
        must_fail(
            lambda: compatible_compare_receipt(receipt, mismatched_manifest),
            "comparison wired policy mismatch",
        )
        old_manifest = json.loads(json.dumps(current_manifest))
        old_manifest.pop("schema_version")
        json_write(receipt / "manifest.json", old_manifest)
        must_fail(
            lambda: compatible_compare_receipt(receipt, current_manifest),
            "old comparison manifest schema",
        )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        occupied_port = int(listener.getsockname()[1])
        must_fail(lambda: assert_port_vacant(occupied_port), "occupied-port preflight")
    finally:
        listener.close()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        print("bench_dsv4_dspark_gpu_markov.py: offline self-test passed")
        return 0
    try:
        out_dir = run_benchmark(args)
    except KeyboardInterrupt:
        print("benchmark interrupted; PID-specific teardown and partial raw receipt were preserved", file=sys.stderr)
        return 130
    except (BenchError, OSError, subprocess.SubprocessError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1
    print(f"benchmark completed authenticated output gate; receipt: {out_dir}")
    print("inspect summary.json for exact covered and missing arms")
    print(f"speed summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
