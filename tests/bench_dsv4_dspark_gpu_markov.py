#!/usr/bin/env python3
"""Proof-grade local benchmark for DSV4 speculative-decoding fast paths.

This intentionally compares separate fresh server boots of the same
ReleaseFast binary and model directory:

  S  plain serial (no ``--dspark``)
  L  legacy DSpark (``--dspark``)
  G  DSpark plus ``MLX_SERVE_DSPARK_GPU_MARKOV_IDS=1``
  J  G plus ``MLX_SERVE_DSPARK_JOIN_VERIFY_EVAL=1``
  B2 GPU-Markov DSpark, ``MLX_SERVE_DSPARK_BLOCK_CAP=2``, fast defaults
  E2 B2 plus exact ``MLX_SERVE_DSV4_M1_BATCH_GEMV=1``
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
The exact batched-M1 comparison is the strict balanced experiment
``--execution-profile fast-default --arms S,B2,E2``.  It runs all six S/B2/E2
permutations on fresh boots, with one excluded same-shape warmup followed by
exactly eight measured requests per boot (up to 128 completion tokens).  It
requires every measured output contract to match across every arm and boot,
requires both implementation-layer engagement markers, and reports B2/E2 plus
S/E2 ratios from that same receipt.  Its convergence gate uses the final three
requests from each boot: every response ``predicted_ms``, metrics TTFT,
response ``prompt_ms``, and wall-time spread must be at most 5% of that
metric's median.  The median denominator is robust to a single high/low sample;
a non-positive median is explicitly nonconvergent.  It also requires
``--libmlx /absolute/path/libmlx.dylib``; the harness pins and audits that same
hashed runtime for S, B2, and E2.

``--m1-lifecycle-diagnostic`` is a deliberately narrower, non-promotion-grade
pilot for the still-default-off E2 candidate.  It accepts only B2/E2 under the
fast defaults, with wired kernels off and one pinned ``libmlx.dylib``.  Each
fresh boot receives exactly one excluded same-shape warmup and five 128-token
measurements.  The harness waits for a genuinely idle server after every one,
captures two settled metrics snapshots plus live ``/props``, and rejects the
receipt if the post-first-measured memory series is unstable.  Optional
``--dspark-profile-every`` applies identically to B2 and E2.  Its timing is
diagnostic only: it is never a promotion-grade speed comparison.
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
  python3 tests/bench_dsv4_dspark_gpu_markov.py --execution-profile fast-default --arms S,B2,E2 --mode balanced --libmlx /tmp/patched-mlx/lib/libmlx.dylib --max-tokens 128
  python3 tests/bench_dsv4_dspark_gpu_markov.py --m1-lifecycle-diagnostic --execution-profile fast-default --mode pilot --arms B2,E2 --wired-mode off --libmlx /tmp/patched-mlx/lib/libmlx.dylib --max-tokens 128

``--self-test`` is offline and exercises only the harness's schedule,
metrics-delta, environment-sanitizing, and log-evidence logic.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


MANIFEST_SCHEMA = "mlx-serve.dsv4-dspark-fast-path-benchmark"
MANIFEST_VERSION = 8
DEFAULT_ARMS = ("S", "L", "G")
LEGACY_ALL_ARMS = ("S", "L", "G", "J")
ALL_ARMS = (*LEGACY_ALL_ARMS, "B2", "R2", "E2")
B2_ARMS = ("S", "B2")
R2_ARMS = ("S", "B2", "R2")
E2_ARMS = ("S", "B2", "E2")
M1_LIFECYCLE_ARMS = ("B2", "E2")
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
E2_BALANCED_ORDERS = (
    ("S", "B2", "E2"),
    ("S", "E2", "B2"),
    ("B2", "S", "E2"),
    ("B2", "E2", "S"),
    ("E2", "S", "B2"),
    ("E2", "B2", "S"),
)
STRICT_E2_MEASURED_PER_BOOT = 8
STRICT_E2_FINAL_STEADY_STATE_REQUESTS = 3
STRICT_E2_MAX_TOKENS = 128
STRICT_E2_CONVERGENCE_MAX_SPREAD_PERCENT = 5.0
M1_LIFECYCLE_MEASURED_PER_BOOT = 5
M1_LIFECYCLE_MAX_TOKENS = 128
M1_LIFECYCLE_SETTLE_SECONDS = 2.2
M1_LIFECYCLE_SETTLE_TIMEOUT_SECONDS = 30.0
M1_LIFECYCLE_POLL_SECONDS = 0.2
M1_LIFECYCLE_MEMORY_TOLERANCE_BYTES = 512 * 1024 * 1024
MIB_BYTES = 1024 * 1024
ARM_NAMES = {
    "S": "plain_serial",
    "L": "legacy_dspark",
    "G": "dspark_gpu_markov_ids",
    "J": "dspark_gpu_markov_ids_joined_verify_eval",
    "B2": "dspark_gpu_markov_ids_block_cap_2_fast_defaults",
    "R2": "dspark_gpu_markov_ids_block_cap_2_replay_commit_fast_defaults",
    "E2": "dspark_gpu_markov_ids_block_cap_2_exact_batched_m1_fast_defaults",
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
M1_BATCH_GEMV_ENV = "MLX_SERVE_DSV4_M1_BATCH_GEMV"
DYLD_LIBRARY_PATH_ENV = "DYLD_LIBRARY_PATH"
DYLD_PRINT_LIBRARIES_ENV = "DYLD_PRINT_LIBRARIES"
DYLD_ENV_PREFIX = "DYLD_"
PINNED_LOADER_ENV_KEYS = (DYLD_LIBRARY_PATH_ENV, DYLD_PRINT_LIBRARIES_ENV)
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
M1_BATCH_GEMV_ZIG_MARKER = (
    "dsv4: exact batched M1 GEMV path engaged (serial reduction over grid.z rows)"
)
M1_BATCH_GEMV_BACKEND_MARKER = (
    "mlx-serve: exact batched M1 GEMV backend engaged (preserved M=1 shared-B batch)"
)
DSPARK_STATS_MARKER = "[spec-stats] mode=dspark"
DSPARK_PROFILE_MARKER = "[dspark-prof]"
LOWER_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DYLD_LIBMLX_LOADED_IMAGE = re.compile(
    r"^dyld\[\d+\]:\s+(?:<[^>]+>\s+)?(?P<path>/.*?/libmlx\.dylib)\s*$"
)

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
        "--libmlx",
        type=Path,
        help=(
            "exact libmlx.dylib loaded by every boot; required for the strict "
            "S,B2,E2 experiment and M1 lifecycle diagnostic"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "balanced"),
        default="pilot",
        help=(
            "pilot=one pass; balanced=six S/L/G counterbalanced orders or "
            "four position-balanced orders when J is selected; fast-default "
            "S,B2 uses its own two-order position-balanced schedule, while "
            "S,B2,R2 uses forward/reverse S,B2,R2 and R2,B2,S orders and the "
            "strict S,B2,E2 experiment uses all six S,B2,E2 permutations; "
            "the M1 lifecycle diagnostic is B2,E2 pilot-only"
        ),
    )
    parser.add_argument(
        "--execution-profile",
        choices=tuple(EXECUTION_PROFILES),
        default="conservative",
        help=(
            "conservative disables three DSV4 fast paths (default); "
            "fast-default is permitted only for explicit S,B2, S,B2,R2, or "
            "balanced S,B2,E2 experiments, plus the authenticated B2,E2 M1 "
            "lifecycle diagnostic; it leaves those kill switches absent after "
            "sanitization"
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
        metavar="S,L,G,J,B2,R2,E2",
        help=(
            "selected/recovery subset, e.g. --arms G or --arms S,J; "
            "S,B2 is the fast-default B2 experiment and S,B2,R2 is its "
            "replay-commit extension; S,B2,E2 is the strict balanced exact-M1 "
            "comparison; B2,E2 is available only with --m1-lifecycle-diagnostic; "
            "default is S,L,G"
        ),
    )
    parser.add_argument(
        "--dspark-profile-every",
        type=int,
        metavar="ROUNDS",
        help=(
            "B2 pilot-only diagnostic, or symmetric B2/E2 M1 lifecycle "
            "diagnostic: enable DSpark phase profiling and report every ROUNDS "
            "(1..1024); profiled timing is explicitly diagnostic"
        ),
    )
    parser.add_argument(
        "--m1-lifecycle-diagnostic",
        action="store_true",
        help=(
            "authenticated non-promotion-grade M1 lifecycle probe; requires "
            "fast-default pilot B2,E2, --libmlx, --wired-mode off, and "
            "--max-tokens 128"
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
        help=(
            "measured requests after the excluded warmup (must be exactly 1, "
            "except strict balanced S,B2,E2 which requires exactly 8 and the "
            "M1 lifecycle diagnostic which requires exactly 5)"
        ),
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
        help=(
            "completion-token cap for warmups and measurements (strict balanced "
            "S,B2,E2 accepts 1..128; M1 lifecycle diagnostic requires 128; "
            "default: %(default)s)"
        ),
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


def runtime_loader_provenance(libmlx: Path | None) -> dict[str, Any]:
    """Describe the intentionally pinned MLX runtime without leaking other env."""
    if libmlx is None:
        return {
            "mode": "unconfigured",
            "libmlx_path": None,
            "libmlx_parent": None,
            "libmlx_sha256": None,
            "server_env": {},
            "sanitized_inherited_keys": [],
            "policy": "legacy loader environment preserved because --libmlx was absent",
        }
    resolved = libmlx.resolve()
    return {
        "mode": "pinned",
        "libmlx_path": str(resolved),
        "libmlx_parent": str(resolved.parent),
        "libmlx_sha256": sha256_file(resolved),
        "server_env": {
            DYLD_LIBRARY_PATH_ENV: str(resolved.parent),
            DYLD_PRINT_LIBRARIES_ENV: "1",
        },
        "sanitized_inherited_keys": sorted(
            key for key in os.environ if key.startswith(DYLD_ENV_PREFIX)
        ),
        "approved_server_env_keys": list(PINNED_LOADER_ENV_KEYS),
        "policy": (
            "remove every inherited DYLD_* key, then set only the approved deterministic "
            "loader env for the same canonical hashed libmlx on every boot"
        ),
    }


def output_contract_sha256(contract: dict[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def is_strict_e2_gate(
    execution_profile: str, selected_arms: tuple[str, ...], mode: str
) -> bool:
    """Whether this invocation is the authenticated E2 convergence gate."""
    return (
        execution_profile == "fast-default"
        and set(selected_arms) == set(E2_ARMS)
        and mode == "balanced"
    )


def is_m1_lifecycle_diagnostic(
    enabled: bool,
    execution_profile: str,
    selected_arms: tuple[str, ...],
    mode: str,
) -> bool:
    """Whether this is the intentionally narrow B2/E2 lifecycle probe."""
    return (
        enabled
        and execution_profile == "fast-default"
        and set(selected_arms) == set(M1_LIFECYCLE_ARMS)
        and mode == "pilot"
    )


def m1_lifecycle_diagnostic_provenance(enabled: bool) -> dict[str, Any]:
    """Receipt policy for a probe that must never be presented as promotion data."""
    return {
        "enabled": enabled,
        "promotion_grade": False,
        "required_arms": list(M1_LIFECYCLE_ARMS) if enabled else [],
        "required_execution_profile": "fast-default" if enabled else None,
        "required_mode": "pilot" if enabled else None,
        "required_wired_mode": "off" if enabled else None,
        "required_max_tokens": M1_LIFECYCLE_MAX_TOKENS if enabled else None,
        "warmups_excluded_per_boot": 1 if enabled else None,
        "measured_requests_per_boot": M1_LIFECYCLE_MEASURED_PER_BOOT if enabled else None,
        "settle_seconds_after_idle": M1_LIFECYCLE_SETTLE_SECONDS if enabled else None,
        "memory_span_tolerance_bytes": (
            M1_LIFECYCLE_MEMORY_TOLERANCE_BYTES if enabled else None
        ),
    }


def required_measured_per_boot(
    execution_profile: str,
    selected_arms: tuple[str, ...],
    mode: str,
    *,
    m1_lifecycle_diagnostic: bool = False,
) -> int:
    """Keep legacy lanes one-shot while making the strict E2 lane repeatable."""
    if is_m1_lifecycle_diagnostic(
        m1_lifecycle_diagnostic, execution_profile, selected_arms, mode
    ):
        return M1_LIFECYCLE_MEASURED_PER_BOOT
    if is_strict_e2_gate(execution_profile, selected_arms, mode):
        return STRICT_E2_MEASURED_PER_BOOT
    return 1


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
        if set(selected_arms) == set(E2_ARMS):
            return E2_BALANCED_ORDERS
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
                    f"--arms accepts only S, L, G, J, B2, R2, and E2; got {arm!r}"
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
        REPLAY_COMMIT_ENV: "0",
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


def e2_extra_env(dspark_profile_every: int | None = None) -> dict[str, str]:
    """E2 is B2 with the exact batched-M1 candidate as its only env delta."""
    return {
        **b2_extra_env(dspark_profile_every),
        M1_BATCH_GEMV_ENV: "1",
    }


def m1_lifecycle_env_delta(dspark_profile_every: int | None) -> dict[str, Any]:
    """Authenticate that B2/E2 differ only by the M1 candidate switch."""
    b2 = b2_extra_env(dspark_profile_every)
    e2 = e2_extra_env(dspark_profile_every)
    added = {key: value for key, value in e2.items() if b2.get(key) != value}
    removed = {key: value for key, value in b2.items() if e2.get(key) != value}
    if added != {M1_BATCH_GEMV_ENV: "1"} or removed:
        raise BenchError(
            "M1 lifecycle B2/E2 environment delta is not exactly the M1 switch: "
            f"added={added}, removed={removed}"
        )
    return {
        "B2_mlx_serve_env": b2,
        "E2_mlx_serve_env": e2,
        "E2_minus_B2": added,
        "B2_minus_E2": removed,
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
    libmlx: Path | None = None,
) -> tuple[dict[str, str], list[str], list[str]]:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}")
    if execution_profile not in EXECUTION_PROFILES:
        raise ValueError(f"unknown execution profile {execution_profile!r}")
    inherited = dict(os.environ)
    removed = sorted(key for key in inherited if key.startswith("MLX_SERVE_"))
    removed_loader = (
        sorted(key for key in inherited if key.startswith(DYLD_ENV_PREFIX))
        if libmlx is not None
        else []
    )
    env = {
        key: value
        for key, value in inherited.items()
        if not key.startswith("MLX_SERVE_")
        and not (libmlx is not None and key.startswith(DYLD_ENV_PREFIX))
    }
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
    if arm == "E2":
        env.update(e2_extra_env(dspark_profile_every))
    if libmlx is not None:
        resolved = libmlx.resolve()
        env[DYLD_LIBRARY_PATH_ENV] = str(resolved.parent)
        env[DYLD_PRINT_LIBRARIES_ENV] = "1"
    return env, removed, removed_loader


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
        "E2": {
            "dspark_server_flag": True,
            "extra_mlx_serve_env": e2_extra_env(dspark_profile_every),
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


def get_props_snapshot(port: int, raw_path: Path) -> dict[str, Any]:
    snapshot = http_json(f"http://127.0.0.1:{port}/props", timeout=15.0)
    if not isinstance(snapshot, dict):
        raise BenchError("/props returned a non-object JSON value")
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


def required_nonnegative_metric(
    snapshot: dict[str, Any], label: str, *keys: str
) -> float:
    value = number_at(snapshot, *keys)
    if value is None or not math.isfinite(value) or value < 0.0:
        raise BenchError(f"lifecycle diagnostic needs non-negative {label}")
    return value


def lifecycle_completion_baseline(
    snapshot: dict[str, Any], request_id: str
) -> dict[str, float]:
    """Counters that prove a completed HTTP response reached server finalization."""
    return {
        "requests_success_total": required_nonnegative_metric(
            snapshot, f"{request_id} counters.requests_success_total", "counters", "requests_success_total"
        ),
        "e2e_request_latency_count": required_nonnegative_metric(
            snapshot,
            f"{request_id} histograms.e2e_request_latency_seconds.count",
            "histograms",
            "e2e_request_latency_seconds",
            "count",
        ),
    }


def lifecycle_metrics_state(
    snapshot: dict[str, Any], baseline: dict[str, float], request_id: str
) -> dict[str, Any]:
    running = required_nonnegative_metric(
        snapshot, f"{request_id} gauges.requests_running", "gauges", "requests_running"
    )
    waiting = required_nonnegative_metric(
        snapshot, f"{request_id} gauges.requests_waiting", "gauges", "requests_waiting"
    )
    prefilling = required_nonnegative_metric(
        snapshot, f"{request_id} gauges.requests_prefilling", "gauges", "requests_prefilling"
    )
    success_total = required_nonnegative_metric(
        snapshot,
        f"{request_id} counters.requests_success_total",
        "counters",
        "requests_success_total",
    )
    e2e_count = required_nonnegative_metric(
        snapshot,
        f"{request_id} histograms.e2e_request_latency_seconds.count",
        "histograms",
        "e2e_request_latency_seconds",
        "count",
    )
    success_updated = success_total >= baseline["requests_success_total"] + 1.0
    histogram_updated = e2e_count >= baseline["e2e_request_latency_count"] + 1.0
    idle = running == 0.0 and waiting == 0.0 and prefilling == 0.0
    return {
        "requests_running": running,
        "requests_waiting": waiting,
        "requests_prefilling": prefilling,
        "requests_success_total": success_total,
        "e2e_request_latency_count": e2e_count,
        "success_counter_updated": success_updated,
        "success_histogram_updated": histogram_updated,
        "idle": idle,
        "ready": idle and success_updated and histogram_updated,
    }


def lifecycle_memory_snapshot(
    metrics: dict[str, Any], props: dict[str, Any], request_id: str
) -> dict[str, Any]:
    """Record all three memory axes without mistaking MLX active bytes for RSS."""
    active = required_nonnegative_metric(
        props, f"{request_id} props.memory.active_bytes", "memory", "active_bytes"
    )
    cache = required_nonnegative_metric(
        props, f"{request_id} props.memory.cache_bytes", "memory", "cache_bytes"
    )
    available = required_nonnegative_metric(
        props, f"{request_id} props.memory.available_bytes", "memory", "available_bytes"
    )
    footprint_mb = required_nonnegative_metric(
        metrics, f"{request_id} metrics.gauges.memory_mb", "gauges", "memory_mb"
    )
    return {
        "active_bytes": active,
        "cache_bytes": cache,
        "active_plus_cache_bytes": active + cache,
        "footprint_mb": footprint_mb,
        "footprint_bytes": footprint_mb * MIB_BYTES,
        "available_bytes": available,
        "footprint_source": "metrics.gauges.memory_mb * 1048576",
    }


def capture_m1_lifecycle_settlement(
    *,
    port: int,
    raw_dir: Path,
    request_id: str,
    baseline: dict[str, float],
) -> dict[str, Any]:
    """Wait for server finalization, then capture a second idle memory snapshot.

    The completion endpoint can return before the scheduler clears its live
    gauges.  This diagnostic deliberately does not treat that early response as
    a stable memory point, and it does not alter the server's memory guard.
    """
    deadline = time.monotonic() + M1_LIFECYCLE_SETTLE_TIMEOUT_SECONDS
    poll_paths: list[str] = []
    last_state: dict[str, Any] | None = None
    settled_metrics: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        poll_path = raw_dir / f"{request_id}.metrics-settle-poll-{len(poll_paths):02d}.json"
        metrics = get_metric_snapshot(port, poll_path)
        poll_paths.append(str(poll_path))
        state = lifecycle_metrics_state(metrics, baseline, request_id)
        last_state = state
        if state["ready"]:
            settled_metrics = metrics
            break
        time.sleep(M1_LIFECYCLE_POLL_SECONDS)
    if settled_metrics is None:
        raise BenchError(
            f"{request_id}: server did not become idle with an updated success "
            f"counter and histogram within {M1_LIFECYCLE_SETTLE_TIMEOUT_SECONDS:.1f}s; "
            f"last state {last_state}"
        )

    delay_started = time.monotonic()
    while (remaining := M1_LIFECYCLE_SETTLE_SECONDS - (time.monotonic() - delay_started)) > 0.0:
        time.sleep(remaining)
    settle_delay_seconds = time.monotonic() - delay_started
    if settle_delay_seconds < M1_LIFECYCLE_SETTLE_SECONDS:
        raise BenchError(f"{request_id}: lifecycle settle delay ended too early")

    delayed_metrics_path = raw_dir / f"{request_id}.metrics-settled.json"
    delayed_metrics = get_metric_snapshot(port, delayed_metrics_path)
    delayed_state = lifecycle_metrics_state(delayed_metrics, baseline, request_id)
    if not delayed_state["ready"]:
        raise BenchError(
            f"{request_id}: server stopped being idle after lifecycle settle delay: {delayed_state}"
        )
    props_path = raw_dir / f"{request_id}.props-settled.json"
    props = get_props_snapshot(port, props_path)
    return {
        "settled": True,
        "baseline": baseline,
        "first_settled_state": lifecycle_metrics_state(settled_metrics, baseline, request_id),
        "second_settled_state": delayed_state,
        "settle_delay_seconds": settle_delay_seconds,
        "memory": lifecycle_memory_snapshot(delayed_metrics, props, request_id),
        "raw": {
            "metrics_settle_polls": poll_paths,
            "metrics_second_settled": str(delayed_metrics_path),
            "props_settled": str(props_path),
        },
    }


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
    m1_batch_gemv_zig_markers = log.count(M1_BATCH_GEMV_ZIG_MARKER)
    m1_batch_gemv_backend_markers = log.count(M1_BATCH_GEMV_BACKEND_MARKER)
    profile_markers = log.count(DSPARK_PROFILE_MARKER)
    evidence = {
        "spec_stats_count": spec_stats,
        "gpu_markov_marker_count": gpu_markers,
        "joined_verify_marker_count": joined_markers,
        "block_cap_b2_marker_count": block_cap_b2_markers,
        "replay_commit_marker_count": replay_commit_markers,
        "m1_batch_gemv_zig_marker_count": m1_batch_gemv_zig_markers,
        "m1_batch_gemv_backend_marker_count": m1_batch_gemv_backend_markers,
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
            or m1_batch_gemv_zig_markers != 0
            or m1_batch_gemv_backend_markers != 0
        ):
            raise BenchError(f"S must have no speculative evidence, observed {evidence}")
    elif arm == "L":
        if (
            spec_stats != expected_dspark_stats
            or gpu_markers != 0
            or joined_markers != 0
            or replay_commit_markers != 0
            or m1_batch_gemv_zig_markers != 0
            or m1_batch_gemv_backend_markers != 0
        ):
            raise BenchError(f"L must have DSpark stats and no fast-path markers, observed {evidence}")
    elif arm == "G":
        if (
            spec_stats != expected_dspark_stats
            or gpu_markers != 1
            or joined_markers != 0
            or replay_commit_markers != 0
            or m1_batch_gemv_zig_markers != 0
            or m1_batch_gemv_backend_markers != 0
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
            or m1_batch_gemv_zig_markers != 0
            or m1_batch_gemv_backend_markers != 0
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
            or m1_batch_gemv_zig_markers != 0
            or m1_batch_gemv_backend_markers != 0
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
            or m1_batch_gemv_zig_markers != 0
            or m1_batch_gemv_backend_markers != 0
            or profile_markers != 0
        ):
            raise BenchError(
                "R2 must have DSpark stats, one GPU marker, one effective=2 cap marker, "
                "one replay-commit marker, no joined marker, and no profile marker; "
                f"observed {evidence}"
            )
    elif arm == "E2":
        if (
            spec_stats != expected_dspark_stats
            or gpu_markers != 1
            or joined_markers != 0
            or block_cap_b2_markers != 1
            or replay_commit_markers != 0
            or m1_batch_gemv_zig_markers != 1
            or m1_batch_gemv_backend_markers != 1
        ):
            raise BenchError(
                "E2 must have DSpark stats, one GPU marker, one effective=2 cap marker, "
                "exactly one Zig and backend batched-M1 marker, and no joined/replay marker; "
                f"observed {evidence}"
            )
        if dspark_profile_required and profile_markers < 1:
            raise BenchError(f"E2 profiled run has no profile marker, observed {evidence}")
        if not dspark_profile_required and profile_markers != 0:
            raise BenchError(f"E2 unprofiled run leaked a profile marker, observed {evidence}")
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return evidence


def assess_runtime_loader(log: str, provenance: dict[str, Any]) -> dict[str, Any]:
    """Require one dyld libmlx image matching the canonical pinned runtime."""
    if provenance.get("mode") != "pinned":
        return {
            "required": False,
            "dyld_libmlx_line_count": 0,
            "expected_path": None,
            "expected_sha256": None,
        }
    expected_raw = provenance.get("libmlx_path")
    expected_sha256 = provenance.get("libmlx_sha256")
    if not isinstance(expected_raw, str) or not isinstance(expected_sha256, str):
        raise BenchError(f"pinned runtime loader provenance is incomplete: {provenance}")
    expected = Path(expected_raw).resolve()
    loaded = [match.group("path") for line in log.splitlines() if (match := DYLD_LIBMLX_LOADED_IMAGE.fullmatch(line))]
    resolved = [str(Path(path).resolve()) for path in loaded]
    evidence = {
        "required": True,
        "dyld_libmlx_line_count": len(loaded),
        "loaded_paths": loaded,
        "resolved_loaded_paths": resolved,
        "expected_path": str(expected),
        "expected_sha256": expected_sha256,
    }
    if len(loaded) != 1:
        raise BenchError(
            "pinned runtime requires exactly one dyld libmlx loaded-image line; "
            f"observed {evidence}"
        )
    if resolved[0] != str(expected):
        raise BenchError(f"dyld loaded the wrong libmlx runtime; observed {evidence}")
    try:
        observed_sha256 = sha256_file(expected)
    except OSError as error:
        raise BenchError(f"cannot re-hash loaded libmlx runtime {expected}: {error}") from error
    evidence["observed_sha256"] = observed_sha256
    if observed_sha256 != expected_sha256:
        raise BenchError(f"loaded libmlx changed after manifest hashing; observed {evidence}")
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
    capture_m1_lifecycle: bool = False,
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
    result = {
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
    if capture_m1_lifecycle:
        baseline = lifecycle_completion_baseline(before, request_id)
        result["lifecycle_settlement"] = capture_m1_lifecycle_settlement(
            port=port,
            raw_dir=raw_dir,
            request_id=request_id,
            baseline=baseline,
        )
    return result


def warmup_request(
    *,
    port: int,
    body: dict[str, Any],
    timeout: float,
    raw_dir: Path,
    request_id: str,
    capture_m1_lifecycle: bool = False,
) -> dict[str, Any] | None:
    before: dict[str, Any] | None = None
    before_path: Path | None = None
    if capture_m1_lifecycle:
        before_path = raw_dir / f"{request_id}.metrics-before.json"
        before = get_metric_snapshot(port, before_path)
    response = http_json(
        f"http://127.0.0.1:{port}/v1/chat/completions", payload=body, timeout=timeout
    )
    if not isinstance(response, dict):
        raise BenchError("warmup completion endpoint returned a non-object JSON value")
    response_path = raw_dir / f"{request_id}.warmup-response.json"
    json_write(response_path, response)
    if not capture_m1_lifecycle:
        return None
    assert before is not None and before_path is not None
    baseline = lifecycle_completion_baseline(before, request_id)
    return {
        "request_id": request_id,
        "succeeded": True,
        "lifecycle_settlement": capture_m1_lifecycle_settlement(
            port=port,
            raw_dir=raw_dir,
            request_id=request_id,
            baseline=baseline,
        ),
        "raw": {
            "response": str(response_path),
            "metrics_before": str(before_path),
        },
    }


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
    runtime_loader: dict[str, Any],
) -> dict[str, Any]:
    # Recheck immediately before Popen as well as in the all-port preflight.
    # This cannot eliminate an OS-level bind race, but it fails cleanly for a
    # stable occupant instead of accepting that process's readiness endpoint.
    assert_port_vacant(port)
    lifecycle_diagnostic = args.m1_lifecycle_diagnostic
    boot_id = f"boot-{boot_index:02d}-{arm}"
    boot_dir = out_dir / "boots" / boot_id
    raw_dir = boot_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    env, removed, removed_loader = sanitized_server_env(
        arm,
        args.execution_profile,
        args.dspark_profile_every,
        args.wired_mode,
        args.libmlx,
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
            "sanitized_inherited_dynamic_loader_keys": removed_loader,
            "server_mlx_serve_env": {key: env[key] for key in sorted(env) if key.startswith("MLX_SERVE_")},
            "server_dynamic_loader_env": {
                key: env[key] for key in PINNED_LOADER_ENV_KEYS if key in env
            },
            "runtime_loader": runtime_loader,
            "m1_lifecycle_diagnostic": m1_lifecycle_diagnostic_provenance(
                lifecycle_diagnostic
            ),
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

    if runtime_loader["mode"] == "pinned":
        observed_libmlx_hash = sha256_file(Path(runtime_loader["libmlx_path"]))
        if observed_libmlx_hash != runtime_loader["libmlx_sha256"]:
            raise BenchError(
                "pinned libmlx changed between manifest creation and server boot: "
                f"expected {runtime_loader['libmlx_sha256']}, got {observed_libmlx_hash}"
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
    if lifecycle_diagnostic:
        result["warmups"] = []
    raised: BaseException | None = None
    try:
        wait_healthy(server, port, args.startup_timeout)
        for warmup_index in range(args.warmups_per_boot):
            warmup = warmup_request(
                port=port,
                body=body,
                timeout=args.request_timeout,
                raw_dir=raw_dir,
                request_id=f"warmup-{warmup_index:02d}",
                capture_m1_lifecycle=lifecycle_diagnostic,
            )
            if lifecycle_diagnostic:
                assert warmup is not None
                result["warmups"].append(warmup)
        for measurement_index in range(measured_per_boot):
            result["measurements"].append(
                measured_request(
                    port=port,
                    body=body,
                    timeout=args.request_timeout,
                    raw_dir=raw_dir,
                    request_id=f"measured-{measurement_index:02d}",
                    capture_m1_lifecycle=lifecycle_diagnostic,
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
                    arm in M1_LIFECYCLE_ARMS and args.dspark_profile_every is not None
                ),
            )
        except BaseException as evidence_error:
            result["engagement_error"] = f"{type(evidence_error).__name__}: {evidence_error}"
            if raised is None:
                raised = evidence_error
        try:
            result["runtime_loader"] = assess_runtime_loader(log, runtime_loader)
        except BaseException as loader_error:
            result["runtime_loader_error"] = f"{type(loader_error).__name__}: {loader_error}"
            if raised is None:
                raised = loader_error
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
        (
            "runtime_loader",
            previous_manifest.get("runtime_loader"),
            current_manifest.get("runtime_loader"),
        ),
        (
            "m1_lifecycle_diagnostic",
            previous_manifest.get("m1_lifecycle_diagnostic"),
            current_manifest.get("m1_lifecycle_diagnostic"),
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


STEADY_STATE_METRICS = (
    "response_predicted_ms",
    "metrics_ttft_ms",
    "response_prompt_ms",
    "wall_seconds",
)


def response_timing_ms(measurement: dict[str, Any], field: str) -> float | None:
    timings = measurement.get("timings")
    if not isinstance(timings, dict):
        return None
    return numeric(timings.get(field))


def metrics_ttft_ms(measurement: dict[str, Any]) -> float | None:
    delta = measurement.get("metrics_histogram_delta")
    if not isinstance(delta, dict):
        return None
    histograms = delta.get("histograms")
    if not isinstance(histograms, dict):
        return None
    ttft = histograms.get("time_to_first_token_seconds")
    if not isinstance(ttft, dict):
        return None
    seconds = numeric(ttft.get("sum_seconds"))
    return None if seconds is None else seconds * 1_000.0


def steady_state_value(measurement: dict[str, Any], metric: str) -> float | None:
    if metric == "response_predicted_ms":
        return response_timing_ms(measurement, "predicted_ms")
    if metric == "metrics_ttft_ms":
        return metrics_ttft_ms(measurement)
    if metric == "response_prompt_ms":
        return response_timing_ms(measurement, "prompt_ms")
    if metric == "wall_seconds":
        return numeric(measurement.get("wall_seconds"))
    raise ValueError(f"unknown steady-state metric {metric!r}")


def robust_spread_summary(values: list[float]) -> dict[str, Any]:
    """Summarize a nonempty metric series using median as the spread denominator.

    ``(max - min) / median`` is deliberately used instead of a minimum or mean
    denominator: it is not destabilized by one unusually low latency and does
    not let a single unusually high latency dilute the reported relative spread.
    A non-positive median has no meaningful percentage denominator, so it is
    preserved as an explicit nonconvergence rather than coerced to zero.
    """
    if not values:
        return {
            "available": False,
            "median": None,
            "minimum": None,
            "maximum": None,
            "spread_percent_of_median": None,
            "converged": False,
            "reason": "no values",
        }
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        return {
            "available": False,
            "median": None,
            "minimum": None,
            "maximum": None,
            "spread_percent_of_median": None,
            "converged": False,
            "reason": "non-finite or negative value",
        }
    median = float(statistics.median(values))
    minimum = min(values)
    maximum = max(values)
    if median <= 0.0:
        return {
            "available": True,
            "median": median,
            "minimum": minimum,
            "maximum": maximum,
            "spread_percent_of_median": None,
            "converged": False,
            "reason": "non-positive median has no robust relative-spread denominator",
        }
    spread_percent = ((maximum - minimum) / median) * 100.0
    return {
        "available": True,
        "median": median,
        "minimum": minimum,
        "maximum": maximum,
        "spread_percent_of_median": spread_percent,
        "converged": spread_percent <= STRICT_E2_CONVERGENCE_MAX_SPREAD_PERCENT,
        "reason": None,
    }


def final_three_steady_state(boot: dict[str, Any]) -> dict[str, Any]:
    """Calculate the strict-gate steady state without discarding bad evidence."""
    measurements = boot.get("measurements")
    if not isinstance(measurements, list):
        measurements = []
    final_three = measurements[-STRICT_E2_FINAL_STEADY_STATE_REQUESTS:]
    sample_count_ok = len(measurements) == STRICT_E2_MEASURED_PER_BOOT
    final_three_complete = len(final_three) == STRICT_E2_FINAL_STEADY_STATE_REQUESTS
    metrics: dict[str, Any] = {}
    for metric in STEADY_STATE_METRICS:
        values = [steady_state_value(row, metric) for row in final_three]
        if not final_three_complete or any(value is None for value in values):
            metrics[metric] = {
                "available": False,
                "median": None,
                "minimum": None,
                "maximum": None,
                "spread_percent_of_median": None,
                "converged": False,
                "reason": "missing final-three measurement value",
            }
        else:
            metrics[metric] = robust_spread_summary(
                [float(value) for value in values if value is not None]
            )
    converged = sample_count_ok and all(metric["converged"] for metric in metrics.values())
    return {
        "measured_requests": len(measurements),
        "required_measured_requests": STRICT_E2_MEASURED_PER_BOOT,
        "final_three_request_ids": [
            row.get("request_id") for row in final_three if isinstance(row, dict)
        ],
        "metrics": metrics,
        "converged": converged,
        "reason": None if converged else "final-three sample count or metric convergence failed",
    }


def strict_e2_convergence_summary(
    boots: list[dict[str, Any]],
    expected_orders: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    """Keep every final-three result, then aggregate same-order arm pairs.

    Each order is a matched fresh-boot block.  Pairwise ratios therefore compare
    arms within the same order before the six order-specific ratios are reduced.
    This keeps order effects visible instead of folding them into an unpaired
    cross-arm average.
    """
    per_boot: list[dict[str, Any]] = []
    by_order: dict[int, list[dict[str, Any]]] = {}
    for boot in boots:
        steady_state = final_three_steady_state(boot)
        record = {
            "boot_id": boot.get("boot_id"),
            "arm": boot.get("arm"),
            "order_index": boot.get("order_index"),
            "order": boot.get("selected_order"),
            "order_position": boot.get("selected_order_position"),
            "steady_state": steady_state,
        }
        per_boot.append(record)
        order_index = boot.get("order_index")
        if isinstance(order_index, int):
            by_order.setdefault(order_index, []).append(record)

    expected_order_strings = ["".join(order) for order in expected_orders]
    paired_by_order: list[dict[str, Any]] = []
    pair_values: dict[str, dict[str, list[float]]] = {
        "B2_vs_S": {metric: [] for metric in STEADY_STATE_METRICS},
        "E2_vs_S": {metric: [] for metric in STEADY_STATE_METRICS},
        "E2_vs_B2": {metric: [] for metric in STEADY_STATE_METRICS},
    }
    pair_definitions = (("B2_vs_S", "B2", "S"), ("E2_vs_S", "E2", "S"), ("E2_vs_B2", "E2", "B2"))
    for order_index, expected_order in enumerate(expected_orders):
        order_records = by_order.get(order_index, [])
        arms = {str(record["arm"]): record for record in order_records}
        pair_ratios: dict[str, Any] = {}
        for pair_name, numerator_arm, denominator_arm in pair_definitions:
            metrics: dict[str, float | None] = {}
            numerator = arms.get(numerator_arm)
            denominator = arms.get(denominator_arm)
            for metric in STEADY_STATE_METRICS:
                numerator_value = (
                    None
                    if numerator is None
                    else numeric(numerator["steady_state"]["metrics"][metric]["median"])
                )
                denominator_value = (
                    None
                    if denominator is None
                    else numeric(denominator["steady_state"]["metrics"][metric]["median"])
                )
                ratio = finite_ratio(numerator_value, denominator_value)
                metrics[metric] = ratio
                if ratio is not None and math.isfinite(ratio) and ratio >= 0.0:
                    pair_values[pair_name][metric].append(ratio)
            pair_ratios[pair_name] = {
                "numerator_arm": numerator_arm,
                "denominator_arm": denominator_arm,
                "direction": "numerator final-three median / denominator final-three median",
                "metrics": metrics,
            }
        paired_by_order.append(
            {
                "order_index": order_index,
                "order": "".join(expected_order),
                "observed_arms": ordered_arms(arms),
                "boot_ids_by_arm": {arm: arms[arm]["boot_id"] for arm in E2_ARMS if arm in arms},
                "pair_ratios": pair_ratios,
            }
        )

    paired_aggregate = {
        pair_name: {
            "direction": "numerator final-three median / denominator final-three median",
            "orders": len(expected_orders),
            "metrics": {
                metric: robust_spread_summary(values)
                for metric, values in metric_values.items()
            },
        }
        for pair_name, metric_values in pair_values.items()
    }
    expected_boot_count = len(expected_orders) * len(E2_ARMS)
    complete_orders = len(by_order) == len(expected_orders) and all(
        set(str(record["arm"]) for record in by_order.get(index, [])) == set(E2_ARMS)
        for index in range(len(expected_orders))
    )
    all_converged = (
        len(boots) == expected_boot_count
        and complete_orders
        and all(record["steady_state"]["converged"] for record in per_boot)
    )
    return {
        "required": True,
        "policy": {
            "measured_requests_per_boot": STRICT_E2_MEASURED_PER_BOOT,
            "final_steady_state_requests": STRICT_E2_FINAL_STEADY_STATE_REQUESTS,
            "max_spread_percent": STRICT_E2_CONVERGENCE_MAX_SPREAD_PERCENT,
            "spread_formula": "(maximum - minimum) / median * 100",
            "denominator": "median of the final three values; non-positive median is nonconvergent",
            "metrics": {
                "response_predicted_ms": "response timings.predicted_ms",
                "metrics_ttft_ms": "metrics time_to_first_token_seconds delta, converted to ms",
                "response_prompt_ms": "response timings.prompt_ms",
                "wall_seconds": "harness monotonic completion wall time",
            },
        },
        "expected_orders": expected_order_strings,
        "expected_boots": expected_boot_count,
        "observed_boots": len(boots),
        "complete_orders": complete_orders,
        "per_boot": per_boot,
        "paired_by_order": paired_by_order,
        "paired_aggregate": paired_aggregate,
        "all_converged": all_converged,
    }


def m1_lifecycle_series_verdict(values: list[float]) -> dict[str, Any]:
    """Bound a post-warmup memory series without concealing a steady ratchet."""
    if len(values) != M1_LIFECYCLE_MEASURED_PER_BOOT - 1:
        return {
            "available": False,
            "samples": len(values),
            "required_samples": M1_LIFECYCLE_MEASURED_PER_BOOT - 1,
            "span_bytes": None,
            "end_minus_start_bytes": None,
            "monotonic_non_decreasing": None,
            "span_within_tolerance": False,
            "no_monotonic_growth_beyond_tolerance": False,
            "passed": False,
            "reason": "missing post-first-measured settled samples",
        }
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        return {
            "available": False,
            "samples": len(values),
            "required_samples": M1_LIFECYCLE_MEASURED_PER_BOOT - 1,
            "span_bytes": None,
            "end_minus_start_bytes": None,
            "monotonic_non_decreasing": None,
            "span_within_tolerance": False,
            "no_monotonic_growth_beyond_tolerance": False,
            "passed": False,
            "reason": "non-finite or negative settled memory value",
        }
    span = max(values) - min(values)
    end_minus_start = values[-1] - values[0]
    monotonic_non_decreasing = all(
        current >= previous for previous, current in zip(values, values[1:])
    )
    span_within_tolerance = span <= M1_LIFECYCLE_MEMORY_TOLERANCE_BYTES
    no_monotonic_growth_beyond_tolerance = not (
        monotonic_non_decreasing
        and end_minus_start > M1_LIFECYCLE_MEMORY_TOLERANCE_BYTES
    )
    passed = span_within_tolerance and no_monotonic_growth_beyond_tolerance
    return {
        "available": True,
        "samples": len(values),
        "required_samples": M1_LIFECYCLE_MEASURED_PER_BOOT - 1,
        "minimum_bytes": min(values),
        "maximum_bytes": max(values),
        "span_bytes": span,
        "end_minus_start_bytes": end_minus_start,
        "monotonic_non_decreasing": monotonic_non_decreasing,
        "span_within_tolerance": span_within_tolerance,
        "no_monotonic_growth_beyond_tolerance": no_monotonic_growth_beyond_tolerance,
        "passed": passed,
        "reason": None if passed else "settled memory span or monotonic growth exceeded tolerance",
    }


def m1_lifecycle_verdict(boots: list[dict[str, Any]]) -> dict[str, Any]:
    """Authenticate lifecycle stability separately from exact-output equivalence.

    The first measured completion is intentionally omitted from the stability
    series because it may pay the one-time pool/page warmup.  It is still
    required to succeed and settle.  The excluded warmup is checked too.
    """
    relevant = [boot for boot in boots if boot.get("arm") in M1_LIFECYCLE_ARMS]
    expected_arms = set(M1_LIFECYCLE_ARMS)
    observed_arms = {str(boot.get("arm")) for boot in relevant}
    expected_boots = len(M1_LIFECYCLE_ARMS)
    per_boot: list[dict[str, Any]] = []
    for boot in relevant:
        warmups = boot.get("warmups")
        measurements = boot.get("measurements")
        if not isinstance(warmups, list):
            warmups = []
        if not isinstance(measurements, list):
            measurements = []
        request_records = [*warmups, *measurements]
        requests_succeeded = (
            boot.get("error") is None
            and len(warmups) == 1
            and len(measurements) == M1_LIFECYCLE_MEASURED_PER_BOOT
            and all(
                isinstance(row, dict)
                and (
                    row.get("succeeded") is True
                    if row.get("request_id", "").startswith("warmup-")
                    else isinstance(row.get("output_contract"), dict)
                )
                for row in request_records
            )
        )
        settlements = [
            row.get("lifecycle_settlement") if isinstance(row, dict) else None
            for row in request_records
        ]
        all_settled = all(
            isinstance(settlement, dict)
            and settlement.get("settled") is True
            and isinstance(settlement.get("second_settled_state"), dict)
            and settlement["second_settled_state"].get("ready") is True
            for settlement in settlements
        )
        post_first_measured = measurements[1:]
        active_plus_cache: list[float] = []
        footprint: list[float] = []
        for row in post_first_measured:
            settlement = row.get("lifecycle_settlement") if isinstance(row, dict) else None
            memory = settlement.get("memory") if isinstance(settlement, dict) else None
            active_cache_value = (
                None
                if not isinstance(memory, dict)
                else numeric(memory.get("active_plus_cache_bytes"))
            )
            footprint_value = (
                None if not isinstance(memory, dict) else numeric(memory.get("footprint_bytes"))
            )
            active_plus_cache.append(
                float("nan") if active_cache_value is None else active_cache_value
            )
            footprint.append(
                float("nan") if footprint_value is None else footprint_value
            )
        active_cache_verdict = m1_lifecycle_series_verdict(active_plus_cache)
        footprint_verdict = m1_lifecycle_series_verdict(footprint)
        passed = (
            requests_succeeded
            and all_settled
            and active_cache_verdict["passed"]
            and footprint_verdict["passed"]
        )
        per_boot.append(
            {
                "boot_id": boot.get("boot_id"),
                "arm": boot.get("arm"),
                "all_requests_succeeded": requests_succeeded,
                "all_requests_settled": all_settled,
                "post_first_measured_request_ids": [
                    row.get("request_id") for row in post_first_measured if isinstance(row, dict)
                ],
                "active_plus_cache": active_cache_verdict,
                "footprint": footprint_verdict,
                "passed": passed,
            }
        )
    complete_boot_set = len(relevant) == expected_boots and observed_arms == expected_arms
    all_passed = complete_boot_set and all(record["passed"] for record in per_boot)
    return {
        "required": True,
        "promotion_grade": False,
        "policy": {
            "warmups_excluded_per_boot": 1,
            "measured_requests_per_boot": M1_LIFECYCLE_MEASURED_PER_BOOT,
            "settle_seconds_after_idle": M1_LIFECYCLE_SETTLE_SECONDS,
            "memory_span_tolerance_bytes": M1_LIFECYCLE_MEMORY_TOLERANCE_BYTES,
            "memory_series": ["active_plus_cache_bytes", "footprint_bytes"],
            "first_measured_request_excluded_from_memory_series": True,
            "requires_idle_gauges": [
                "requests_running",
                "requests_waiting",
                "requests_prefilling",
            ],
            "requires_completion_updates": [
                "counters.requests_success_total",
                "histograms.e2e_request_latency_seconds.count",
            ],
        },
        "expected_arms": list(M1_LIFECYCLE_ARMS),
        "observed_arms": ordered_arms(observed_arms),
        "expected_boots": expected_boots,
        "observed_boots": len(relevant),
        "complete_boot_set": complete_boot_set,
        "per_boot": per_boot,
        "passed": all_passed,
    }


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
    # The explicit fast-default B2/R2/E2 experiments, and the authenticated
    # B2/E2 lifecycle diagnostic, have their own canonical baselines.
    # Every other run retains S/L/G as the canonical baseline, with selected
    # extension arms additionally required for its requested result.
    selected_set = set(selected_arms)
    if selected_set == set(M1_LIFECYCLE_ARMS):
        canonical_baseline = M1_LIFECYCLE_ARMS
    elif selected_set == set(E2_ARMS):
        canonical_baseline = E2_ARMS
    elif selected_set == set(R2_ARMS):
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
        set(selected_arms) == set(E2_ARMS)
        and expected_orders == E2_BALANCED_ORDERS
        and frozen_complete
    ):
        label = "complete frozen six-order counterbalanced S/B2/E2 schedule"
        counterbalanced = True
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
    execution_profile: str = "conservative",
    boots: list[dict[str, Any]] | None = None,
    m1_lifecycle_diagnostic: bool = False,
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
    strict_e2 = is_strict_e2_gate(execution_profile, selected_arms, mode)
    lifecycle_diagnostic = is_m1_lifecycle_diagnostic(
        m1_lifecycle_diagnostic, execution_profile, selected_arms, mode
    )
    if strict_e2:
        if boots is None:
            steady_state_convergence: dict[str, Any] = {
                "required": True,
                "all_converged": False,
                "reason": "strict E2 boot records were not provided",
            }
        else:
            steady_state_convergence = strict_e2_convergence_summary(boots, expected_orders)
    else:
        steady_state_convergence = {
            "required": False,
            "all_converged": None,
            "reason": "final-three convergence is required only for strict balanced S/B2/E2",
        }
    if lifecycle_diagnostic:
        lifecycle_verdict = (
            m1_lifecycle_verdict(boots)
            if boots is not None
            else {
                "required": True,
                "promotion_grade": False,
                "passed": False,
                "reason": "M1 lifecycle boot records were not provided",
            }
        )
    else:
        lifecycle_verdict = {
            "required": False,
            "promotion_grade": False,
            "passed": None,
            "reason": "lifecycle verdict is required only for --m1-lifecycle-diagnostic",
        }
    if coverage["complete_for_required_arms"]:
        cross_arm = "exact match across all required arms: " + ",".join(coverage["required_arms"])
    else:
        cross_arm = (
            "exact match only across covered arms "
            + ",".join(coverage["covered_arms"])
            + "; missing required arms "
            + ",".join(coverage["missing_required_arms"])
        )
    if lifecycle_diagnostic:
        speed_scope = (
            "authenticated B2/E2 M1 lifecycle diagnostic; timing is diagnostic only, "
            "not promotion-grade"
        )
    elif mode == "pilot":
        speed_scope = "same-receipt one-pass pilot; directional timing only"
    elif schedule["counterbalanced"] and not strict_e2:
        speed_scope = "same-receipt complete counterbalanced S/L/G timing comparison"
    elif (
        set(selected_arms) == set(B2_ARMS)
        and expected_orders == B2_BALANCED_ORDERS
        and schedule["position_balanced"]
    ):
        speed_scope = "same-receipt complete position-balanced S/B2 timing comparison"
    elif (
        strict_e2
        and expected_orders == E2_BALANCED_ORDERS
        and schedule["complete_frozen_schedule"]
    ):
        if steady_state_convergence["all_converged"]:
            speed_scope = "same-receipt complete counterbalanced S/B2/E2 timing comparison"
        else:
            speed_scope = (
                "strict S/B2/E2 output gate passed but final-three timing convergence failed; "
                "timing comparison is withheld"
            )
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
        profile_arms = "B2/E2" if lifecycle_diagnostic else "B2"
        speed_scope = (
            f"{profile_arms} phase profiling every {dspark_profile_every} round(s); "
            "timing is diagnostic, not a symmetric speed comparison; "
            + speed_scope
        )
    return {
        "output_contract_sha256": output_hash,
        "output_arm_coverage": coverage,
        "schedule": schedule,
        "steady_state_convergence": steady_state_convergence,
        "m1_lifecycle_verdict": lifecycle_verdict,
        "comparability": {
            "same_binary": True,
            "same_model_directory": True,
            "same_prompt": True,
            "effective_request_config": request_config,
            "speed_reported_only_after_exact_output_contract_match": True,
            "selected_arms": list(selected_arms),
            "b2_dspark_profile_every": dspark_profile_every,
            "m1_lifecycle_diagnostic": m1_lifecycle_diagnostic_provenance(
                lifecycle_diagnostic
            ),
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
            "B2_vs_E2": arm_metric_ratio(arms, "B2", "E2"),
            "S_vs_E2": arm_metric_ratio(arms, "S", "E2"),
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
    libmlx: Path | None = None,
    *,
    m1_lifecycle_diagnostic: bool = False,
) -> None:
    selected = set(selected_arms)
    is_b2_pair = selected == set(B2_ARMS)
    is_r2_triplet = selected == set(R2_ARMS)
    is_e2_triplet = selected == set(E2_ARMS)
    is_lifecycle_pair = selected == set(M1_LIFECYCLE_ARMS)
    if execution_profile == "fast-default" and not (
        is_b2_pair or is_r2_triplet or is_e2_triplet or is_lifecycle_pair
    ):
        raise BenchError(
            "--execution-profile fast-default is reserved for the explicit "
            "--arms S,B2, --arms S,B2,R2, balanced --arms S,B2,E2, or the "
            "authenticated B2,E2 M1 lifecycle diagnostic"
        )
    if "R2" in selected and not is_r2_triplet:
        raise BenchError("R2 requires the explicit replay comparison --arms S,B2,R2")
    if "E2" in selected and not (is_e2_triplet or (m1_lifecycle_diagnostic and is_lifecycle_pair)):
        raise BenchError("E2 requires the exact comparison --arms S,B2,E2")
    if is_e2_triplet and mode != "balanced":
        raise BenchError("the S,B2,E2 exact comparison requires --mode balanced")
    if is_e2_triplet and libmlx is None:
        raise BenchError("the S,B2,E2 exact comparison requires --libmlx PATH")
    if "B2" in selected and not (
        is_b2_pair
        or is_r2_triplet
        or is_e2_triplet
        or (m1_lifecycle_diagnostic and is_lifecycle_pair)
    ):
        raise BenchError("B2 requires --arms S,B2, --arms S,B2,R2, or --arms S,B2,E2")
    if ("B2" in selected or "R2" in selected or "E2" in selected) and execution_profile != "fast-default":
        raise BenchError("B2, R2, and E2 require --execution-profile fast-default")
    if dspark_profile_every is not None:
        if not (is_b2_pair or (m1_lifecycle_diagnostic and is_lifecycle_pair)):
            raise BenchError(
                "--dspark-profile-every is available only for S,B2 or the B2,E2 "
                "M1 lifecycle diagnostic"
            )
        if not 1 <= dspark_profile_every <= 1024:
            raise BenchError("--dspark-profile-every must be in 1..1024")
        if mode != "pilot":
            raise BenchError(
                "--dspark-profile-every is pilot-only; balanced speed receipts must be unprofiled"
            )


def validate_m1_lifecycle_diagnostic_contract(
    *,
    enabled: bool,
    execution_profile: str,
    selected_arms: tuple[str, ...],
    mode: str,
    wired_mode: str,
    libmlx: Path | None,
    max_tokens: int,
) -> None:
    if not enabled:
        return
    if not is_m1_lifecycle_diagnostic(enabled, execution_profile, selected_arms, mode):
        raise BenchError(
            "--m1-lifecycle-diagnostic requires --execution-profile fast-default "
            "--mode pilot --arms B2,E2"
        )
    if libmlx is None:
        raise BenchError("--m1-lifecycle-diagnostic requires --libmlx PATH")
    if wired_mode != "off":
        raise BenchError("--m1-lifecycle-diagnostic requires --wired-mode off")
    if max_tokens != M1_LIFECYCLE_MAX_TOKENS:
        raise BenchError(
            f"--m1-lifecycle-diagnostic requires --max-tokens {M1_LIFECYCLE_MAX_TOKENS}"
        )


def validate_sampling_contract(
    execution_profile: str,
    selected_arms: tuple[str, ...],
    mode: str,
    warmups_per_boot: int,
    measured_per_boot: int | None,
    max_tokens: int,
    *,
    m1_lifecycle_diagnostic: bool = False,
) -> int:
    """Validate the fixed same-shape sample counts before any server starts."""
    if warmups_per_boot != 1:
        raise BenchError(
            "--warmups-per-boot must be exactly 1: every fresh boot has one excluded same-shape warmup"
        )
    required_measurements = required_measured_per_boot(
        execution_profile,
        selected_arms,
        mode,
        m1_lifecycle_diagnostic=m1_lifecycle_diagnostic,
    )
    if measured_per_boot is not None and measured_per_boot != required_measurements:
        raise BenchError(
            f"--measured-per-boot must be exactly {required_measurements} per fresh boot "
            "for this experiment"
        )
    if max_tokens < 1:
        raise BenchError("--max-tokens must be at least 1")
    if is_strict_e2_gate(execution_profile, selected_arms, mode) and (
        max_tokens > STRICT_E2_MAX_TOKENS
    ):
        raise BenchError(
            f"strict S,B2,E2 supports --max-tokens in 1..{STRICT_E2_MAX_TOKENS}"
        )
    return required_measurements


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
        args.libmlx,
        m1_lifecycle_diagnostic=args.m1_lifecycle_diagnostic,
    )
    validate_m1_lifecycle_diagnostic_contract(
        enabled=args.m1_lifecycle_diagnostic,
        execution_profile=args.execution_profile,
        selected_arms=selected_arms,
        mode=args.mode,
        wired_mode=args.wired_mode,
        libmlx=args.libmlx,
        max_tokens=args.max_tokens,
    )
    validate_sampling_contract(
        args.execution_profile,
        selected_arms,
        args.mode,
        args.warmups_per_boot,
        args.measured_per_boot,
        args.max_tokens,
        m1_lifecycle_diagnostic=args.m1_lifecycle_diagnostic,
    )
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise BenchError("timeouts must be positive")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise BenchError(f"ReleaseFast binary is not executable: {args.binary}")
    if not args.model.is_dir():
        raise BenchError(f"model directory does not exist: {args.model}")
    if args.libmlx is not None:
        if args.libmlx.name != "libmlx.dylib":
            raise BenchError(f"--libmlx must name libmlx.dylib, got: {args.libmlx}")
        if not args.libmlx.is_file():
            raise BenchError(f"--libmlx is not a file: {args.libmlx}")
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
    if args.libmlx is not None:
        args.libmlx = args.libmlx.resolve()
    if args.compare_receipt is not None:
        args.compare_receipt = args.compare_receipt.resolve()
    selected_arms = parse_selected_arms(args.arms)
    schedule = schedule_for(args.mode, selected_arms)
    selected_schedule = filtered_schedule(schedule, selected_arms)
    validate_args(args, selected_arms, selected_schedule)
    prompt = resolve_prompt(args)
    if not prompt.strip():
        raise BenchError("fixed benchmark prompt is empty")
    measured_per_boot = (
        required_measured_per_boot(
            args.execution_profile,
            selected_arms,
            args.mode,
            m1_lifecycle_diagnostic=args.m1_lifecycle_diagnostic,
        )
        if args.measured_per_boot is None
        else args.measured_per_boot
    )

    out_dir = prepare_out_dir(args)
    request_config = effective_request_config(args.model, args.max_tokens)
    runtime_loader = runtime_loader_provenance(args.libmlx)
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
        "runtime_loader": runtime_loader,
        "effective_arm_config": effective_arm_config(args.dspark_profile_every),
        "m1_lifecycle_diagnostic": m1_lifecycle_diagnostic_provenance(
            args.m1_lifecycle_diagnostic
        ),
    }
    if args.m1_lifecycle_diagnostic:
        run_manifest["m1_lifecycle_env_delta"] = m1_lifecycle_env_delta(
            args.dspark_profile_every
        )
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
                    runtime_loader=runtime_loader,
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
        execution_profile=args.execution_profile,
        boots=boots,
        m1_lifecycle_diagnostic=args.m1_lifecycle_diagnostic,
    )
    summary["receipt_dir"] = str(out_dir)
    json_write(out_dir / "summary.json", summary)
    if (
        is_strict_e2_gate(args.execution_profile, selected_arms, args.mode)
        and not summary["steady_state_convergence"]["all_converged"]
    ):
        raise BenchError(
            "strict S/B2/E2 final-three timing convergence failed; "
            f"inspect preserved summary: {out_dir / 'summary.json'}"
        )
    if args.m1_lifecycle_diagnostic and not summary["m1_lifecycle_verdict"]["passed"]:
        raise BenchError(
            "M1 lifecycle diagnostic failed its settled-memory verdict; inspect preserved summary: "
            f"{out_dir / 'summary.json'}"
        )
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
    assert not default_args.m1_lifecycle_diagnostic
    assert parse_args(["--m1-lifecycle-diagnostic"]).m1_lifecycle_diagnostic
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
    assert "S,B2,E2" in help_output.getvalue()
    assert "--m1-lifecycle-diagnostic" in help_output.getvalue()
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
    assert schedule_for("balanced", E2_ARMS) == E2_BALANCED_ORDERS
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
    assert E2_BALANCED_ORDERS == (
        ("S", "B2", "E2"),
        ("S", "E2", "B2"),
        ("B2", "S", "E2"),
        ("B2", "E2", "S"),
        ("E2", "S", "B2"),
        ("E2", "B2", "S"),
    )
    for arm in E2_ARMS:
        positions = [order.index(arm) for order in E2_BALANCED_ORDERS]
        assert sorted(positions) == [0, 0, 1, 1, 2, 2]
    assert parse_selected_arms(None) == DEFAULT_ARMS
    assert parse_selected_arms(["G"]) == ("G",)
    assert parse_selected_arms(["S,J"]) == ("S", "J")
    assert parse_selected_arms(["S,L", "G"]) == DEFAULT_ARMS
    assert parse_selected_arms(["S,B2"]) == B2_ARMS
    assert parse_selected_arms(["S,B2,R2"]) == R2_ARMS
    assert parse_selected_arms(["S,B2,E2"]) == E2_ARMS
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
    assert filtered_schedule(schedule_for("balanced", E2_ARMS), E2_ARMS) == tuple(
        (order, order) for order in E2_BALANCED_ORDERS
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
    e2_label = schedule_assessment(
        "balanced", E2_ARMS, E2_BALANCED_ORDERS, E2_BALANCED_ORDERS, True
    )
    assert e2_label["label"] == "complete frozen six-order counterbalanced S/B2/E2 schedule"
    assert e2_label["position_balanced"] and e2_label["counterbalanced"]
    pilot_body = request_body(Path("/models/fake"), "prompt", "J", 16)
    assert pilot_body["max_tokens"] == 16 and pilot_body["enable_mtp"] is True

    inherited_keys = (
        "MLX_SERVE_DSPARK_SERIAL_VERIFY",
        BLOCK_CAP_ENV,
        DSPARK_PROFILE_ENV,
        DSPARK_PROFILE_EVERY_ENV,
        REPLAY_COMMIT_ENV,
        M1_BATCH_GEMV_ENV,
        WIRED_ENV,
        "MLX_SERVE_DSV4_DEC_CHAIN",
        DYLD_LIBRARY_PATH_ENV,
        DYLD_PRINT_LIBRARIES_ENV,
        "DYLD_INSERT_LIBRARIES",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_VERSIONED_LIBRARY_PATH",
        "DYLD_IMAGE_SUFFIX",
    )
    previous = {key: os.environ.get(key) for key in inherited_keys}
    os.environ["MLX_SERVE_DSPARK_SERIAL_VERIFY"] = "1"
    os.environ[BLOCK_CAP_ENV] = "5"
    os.environ[DSPARK_PROFILE_ENV] = "1"
    os.environ[DSPARK_PROFILE_EVERY_ENV] = "99"
    os.environ[REPLAY_COMMIT_ENV] = "0"
    os.environ[M1_BATCH_GEMV_ENV] = "0"
    os.environ[WIRED_ENV] = "on"
    os.environ["MLX_SERVE_DSV4_DEC_CHAIN"] = "0"
    os.environ[DYLD_LIBRARY_PATH_ENV] = "/inherited/wrong-mlx"
    os.environ[DYLD_PRINT_LIBRARIES_ENV] = "0"
    os.environ["DYLD_INSERT_LIBRARIES"] = "/inherited/injected.dylib"
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/inherited/fallback"
    os.environ["DYLD_VERSIONED_LIBRARY_PATH"] = "/inherited/versioned"
    os.environ["DYLD_IMAGE_SUFFIX"] = "_debug"
    try:
        legacy, removed, legacy_loader_removed = sanitized_server_env("L")
        gpu, _, _ = sanitized_server_env("G")
        joined, _, _ = sanitized_server_env("J")
        fast_serial, fast_removed, fast_loader_removed = sanitized_server_env(
            "S", "fast-default"
        )
        b2, _, _ = sanitized_server_env("B2", "fast-default")
        profiled_b2, _, _ = sanitized_server_env("B2", "fast-default", 1)
        r2, _, _ = sanitized_server_env("R2", "fast-default")
        assert "MLX_SERVE_DSPARK_SERIAL_VERIFY" in removed
        assert BLOCK_CAP_ENV in fast_removed
        assert DSPARK_PROFILE_ENV in fast_removed
        assert REPLAY_COMMIT_ENV in fast_removed
        assert M1_BATCH_GEMV_ENV in fast_removed
        assert WIRED_ENV in fast_removed
        assert legacy_loader_removed == [] and fast_loader_removed == []
        assert legacy[DYLD_LIBRARY_PATH_ENV] == "/inherited/wrong-mlx"
        assert fast_serial[DYLD_PRINT_LIBRARIES_ENV] == "0"
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
            REPLAY_COMMIT_ENV: "0",
        }
        assert {
            key: value for key, value in profiled_b2.items() if key.startswith("MLX_SERVE_")
        } == {
            GPU_MARKOV_ENV: "1",
            BLOCK_CAP_ENV: "2",
            REPLAY_COMMIT_ENV: "0",
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
            profile = "fast-default" if arm in ("B2", "R2", "E2") else "conservative"
            default_wired, default_removed, _ = sanitized_server_env(
                arm, profile, wired_mode="default"
            )
            off_wired, off_removed, _ = sanitized_server_env(
                arm, profile, wired_mode="off"
            )
            assert WIRED_ENV in default_removed and WIRED_ENV in off_removed
            assert WIRED_ENV not in default_wired
            assert off_wired[WIRED_ENV] == "off"
        with tempfile.TemporaryDirectory() as loader_temp:
            loader_dir = Path(loader_temp)
            (loader_dir / "nested").mkdir()
            loader_path = loader_dir / "libmlx.dylib"
            loader_path.write_bytes(b"deterministic fake mlx runtime")
            noncanonical_loader = loader_dir / "nested" / ".." / "libmlx.dylib"
            loader_provenance = runtime_loader_provenance(noncanonical_loader)
            assert loader_provenance["libmlx_path"] == str(loader_path.resolve())
            assert loader_provenance["libmlx_parent"] == str(loader_dir.resolve())
            assert loader_provenance["libmlx_sha256"] == sha256_file(loader_path)
            assert loader_provenance["server_env"] == {
                DYLD_LIBRARY_PATH_ENV: str(loader_dir.resolve()),
                DYLD_PRINT_LIBRARIES_ENV: "1",
            }
            expected_removed_dyld = sorted(
                key for key in inherited_keys if key.startswith(DYLD_ENV_PREFIX)
            )
            assert loader_provenance["sanitized_inherited_keys"] == expected_removed_dyld
            for arm in E2_ARMS:
                pinned, _, pinned_loader_removed = sanitized_server_env(
                    arm, "fast-default", libmlx=noncanonical_loader
                )
                assert pinned_loader_removed == expected_removed_dyld
                assert {
                    key: pinned[key] for key in PINNED_LOADER_ENV_KEYS
                } == loader_provenance["server_env"]
                assert sorted(key for key in pinned if key.startswith(DYLD_ENV_PREFIX)) == sorted(
                    PINNED_LOADER_ENV_KEYS
                )
                assert pinned.get(M1_BATCH_GEMV_ENV) == ("1" if arm == "E2" else None)
            assert runtime_loader_provenance(None)["mode"] == "unconfigured"
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
        REPLAY_COMMIT_ENV: "0",
    }
    assert effective_arm_config(8)["B2"]["extra_mlx_serve_env"][DSPARK_PROFILE_EVERY_ENV] == "8"
    assert effective_arm_config()["R2"]["extra_mlx_serve_env"] == {
        GPU_MARKOV_ENV: "1",
        BLOCK_CAP_ENV: "2",
        REPLAY_COMMIT_ENV: "1",
    }
    assert effective_arm_config()["E2"]["extra_mlx_serve_env"] == {
        GPU_MARKOV_ENV: "1",
        BLOCK_CAP_ENV: "2",
        REPLAY_COMMIT_ENV: "0",
        M1_BATCH_GEMV_ENV: "1",
    }
    profiled_e2 = effective_arm_config(8)["E2"]["extra_mlx_serve_env"]
    assert profiled_e2 == {
        GPU_MARKOV_ENV: "1",
        BLOCK_CAP_ENV: "2",
        REPLAY_COMMIT_ENV: "0",
        DSPARK_PROFILE_ENV: "1",
        DSPARK_PROFILE_EVERY_ENV: "8",
        M1_BATCH_GEMV_ENV: "1",
    }
    lifecycle_env_delta = m1_lifecycle_env_delta(8)
    assert lifecycle_env_delta["E2_minus_B2"] == {M1_BATCH_GEMV_ENV: "1"}
    assert lifecycle_env_delta["B2_minus_E2"] == {}

    validate_experiment_contract("conservative", DEFAULT_ARMS, "balanced", None)
    validate_experiment_contract("fast-default", B2_ARMS, "pilot", None)
    validate_experiment_contract("fast-default", B2_ARMS, "pilot", 1)
    validate_experiment_contract("fast-default", R2_ARMS, "balanced", None)
    validate_experiment_contract(
        "fast-default", E2_ARMS, "balanced", None, Path("/tmp/libmlx.dylib")
    )
    validate_experiment_contract(
        "fast-default",
        M1_LIFECYCLE_ARMS,
        "pilot",
        1,
        Path("/tmp/libmlx.dylib"),
        m1_lifecycle_diagnostic=True,
    )
    validate_m1_lifecycle_diagnostic_contract(
        enabled=True,
        execution_profile="fast-default",
        selected_arms=M1_LIFECYCLE_ARMS,
        mode="pilot",
        wired_mode="off",
        libmlx=Path("/tmp/libmlx.dylib"),
        max_tokens=128,
    )
    assert is_strict_e2_gate("fast-default", E2_ARMS, "balanced")
    assert not is_strict_e2_gate("conservative", E2_ARMS, "balanced")
    assert required_measured_per_boot("fast-default", E2_ARMS, "balanced") == 8
    assert (
        required_measured_per_boot(
            "fast-default",
            M1_LIFECYCLE_ARMS,
            "pilot",
            m1_lifecycle_diagnostic=True,
        )
        == M1_LIFECYCLE_MEASURED_PER_BOOT
    )
    assert required_measured_per_boot("fast-default", B2_ARMS, "balanced") == 1
    assert (
        validate_sampling_contract("fast-default", E2_ARMS, "balanced", 1, None, 128)
        == STRICT_E2_MEASURED_PER_BOOT
    )
    assert (
        validate_sampling_contract("fast-default", E2_ARMS, "balanced", 1, 8, 128)
        == STRICT_E2_MEASURED_PER_BOOT
    )
    assert validate_sampling_contract("conservative", DEFAULT_ARMS, "pilot", 1, None, 64) == 1
    assert (
        validate_sampling_contract(
            "fast-default",
            M1_LIFECYCLE_ARMS,
            "pilot",
            1,
            5,
            128,
            m1_lifecycle_diagnostic=True,
        )
        == M1_LIFECYCLE_MEASURED_PER_BOOT
    )
    must_fail(
        lambda: validate_m1_lifecycle_diagnostic_contract(
            enabled=True,
            execution_profile="fast-default",
            selected_arms=M1_LIFECYCLE_ARMS,
            mode="balanced",
            wired_mode="off",
            libmlx=Path("/tmp/libmlx.dylib"),
            max_tokens=128,
        ),
        "M1 lifecycle balanced mode",
    )
    must_fail(
        lambda: validate_m1_lifecycle_diagnostic_contract(
            enabled=True,
            execution_profile="fast-default",
            selected_arms=M1_LIFECYCLE_ARMS,
            mode="pilot",
            wired_mode="default",
            libmlx=Path("/tmp/libmlx.dylib"),
            max_tokens=128,
        ),
        "M1 lifecycle wired default",
    )
    must_fail(
        lambda: validate_m1_lifecycle_diagnostic_contract(
            enabled=True,
            execution_profile="fast-default",
            selected_arms=M1_LIFECYCLE_ARMS,
            mode="pilot",
            wired_mode="off",
            libmlx=None,
            max_tokens=128,
        ),
        "M1 lifecycle missing libmlx",
    )
    must_fail(
        lambda: validate_sampling_contract(
            "fast-default",
            M1_LIFECYCLE_ARMS,
            "pilot",
            1,
            1,
            128,
            m1_lifecycle_diagnostic=True,
        ),
        "M1 lifecycle wrong measured request count",
    )
    must_fail(
        lambda: validate_sampling_contract("fast-default", E2_ARMS, "balanced", 1, 5, 128),
        "strict E2 five measured requests",
    )
    must_fail(
        lambda: validate_sampling_contract("fast-default", E2_ARMS, "balanced", 1, 8, 129),
        "strict E2 max-tokens above 128",
    )
    must_fail(
        lambda: validate_sampling_contract("fast-default", E2_ARMS, "balanced", 2, 8, 128),
        "strict E2 extra warmup",
    )
    must_fail(
        lambda: validate_experiment_contract("fast-default", E2_ARMS, "balanced", None),
        "E2 missing --libmlx",
    )
    must_fail(
        lambda: validate_experiment_contract("conservative", B2_ARMS, "pilot", None),
        "B2 conservative profile",
    )
    must_fail(
        lambda: validate_experiment_contract("conservative", R2_ARMS, "balanced", None),
        "R2 conservative profile",
    )
    must_fail(
        lambda: validate_experiment_contract("conservative", E2_ARMS, "balanced", None),
        "E2 conservative profile",
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
        lambda: validate_experiment_contract("fast-default", ("S", "E2"), "balanced", None),
        "E2 missing B2 control",
    )
    must_fail(
        lambda: validate_experiment_contract("fast-default", E2_ARMS, "pilot", None),
        "E2 pilot is not a balanced receipt",
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
    e2_log = (
        f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{BLOCK_CAP_B2_MARKER}\n"
        f"{M1_BATCH_GEMV_ZIG_MARKER}\n{M1_BATCH_GEMV_BACKEND_MARKER}"
    )
    e2_evidence = assess_engagement("E2", e2_log, 1)
    assert e2_evidence["m1_batch_gemv_zig_marker_count"] == 1
    assert e2_evidence["m1_batch_gemv_backend_marker_count"] == 1
    assess_engagement("E2", f"{e2_log}\n{DSPARK_PROFILE_MARKER}", 1, dspark_profile_required=True)
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
    must_fail(
        lambda: assess_engagement(
            "E2",
            f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{BLOCK_CAP_B2_MARKER}\n"
            f"{M1_BATCH_GEMV_BACKEND_MARKER}",
            1,
        ),
        "E2 Zig marker omission",
    )
    must_fail(
        lambda: assess_engagement(
            "E2",
            f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{BLOCK_CAP_B2_MARKER}\n"
            f"{M1_BATCH_GEMV_ZIG_MARKER}",
            1,
        ),
        "E2 backend marker omission",
    )
    must_fail(
        lambda: assess_engagement("E2", f"{e2_log}\n{M1_BATCH_GEMV_BACKEND_MARKER}", 1),
        "E2 duplicate backend marker",
    )
    must_fail(
        lambda: assess_engagement(
            "B2",
            f"{DSPARK_STATS_MARKER}\n{GPU_MARKOV_MARKER}\n{BLOCK_CAP_B2_MARKER}\n"
            f"{M1_BATCH_GEMV_ZIG_MARKER}\n{M1_BATCH_GEMV_BACKEND_MARKER}",
            1,
        ),
        "B2 exact-M1 marker leak",
    )
    with tempfile.TemporaryDirectory() as loader_evidence_temp:
        expected_libmlx = Path(loader_evidence_temp) / "libmlx.dylib"
        expected_libmlx.write_bytes(b"loader evidence fixture")
        loader_provenance = runtime_loader_provenance(expected_libmlx)
        correct_dyld_line = f"dyld[123]: <ABCDEF> {expected_libmlx.resolve()}"
        loader_evidence = assess_runtime_loader(correct_dyld_line, loader_provenance)
        assert loader_evidence["dyld_libmlx_line_count"] == 1
        assert loader_evidence["resolved_loaded_paths"] == [str(expected_libmlx.resolve())]
        assert loader_evidence["observed_sha256"] == sha256_file(expected_libmlx)
        must_fail(
            lambda: assess_runtime_loader("server log without dyld evidence", loader_provenance),
            "missing dyld libmlx evidence",
        )
        must_fail(
            lambda: assess_runtime_loader(
                "dyld[123]: <ABCDEF> /tmp/wrong/libmlx.dylib", loader_provenance
            ),
            "wrong dyld libmlx evidence",
        )
        must_fail(
            lambda: assess_runtime_loader(
                f"{correct_dyld_line}\n{correct_dyld_line}", loader_provenance
            ),
            "duplicate dyld libmlx evidence",
        )
        assert not assess_runtime_loader("legacy log", runtime_loader_provenance(None))[
            "required"
        ]

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
    assert (
        assert_exact_output_equivalence(
            [measurement("S"), measurement("B2"), measurement("E2")]
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
    e2_match = output_arm_coverage(
        [measurement("B2"), measurement("E2")], E2_ARMS, {"arms": ["S"]}
    )
    assert e2_match["canonical_baseline_arms"] == ["S", "B2", "E2"]
    assert e2_match["covered_arms"] == ["S", "B2", "E2"]
    assert e2_match["complete_for_required_arms"]
    lifecycle_match = output_arm_coverage(
        [measurement("B2"), measurement("E2")], M1_LIFECYCLE_ARMS, None
    )
    assert lifecycle_match["canonical_baseline_arms"] == ["B2", "E2"]
    assert lifecycle_match["covered_arms"] == ["B2", "E2"]
    assert lifecycle_match["missing_required_arms"] == []
    assert lifecycle_match["complete_for_required_arms"]

    def speed_measurement(
        arm: str,
        *,
        wall_seconds: float = 1.0,
        ttft_seconds: float = 0.1,
        predicted_ms: float = 900.0,
        prompt_ms: float = 100.0,
    ) -> dict[str, Any]:
        return {
            **measurement(arm),
            "wall_seconds": wall_seconds,
            "metrics_histogram_delta": {
                "histograms": {
                    "time_to_first_token_seconds": {"sum_seconds": ttft_seconds},
                    "decode_time_seconds": {"sum_seconds": wall_seconds - ttft_seconds},
                }
            },
            "timings": {"predicted_ms": predicted_ms, "prompt_ms": prompt_ms},
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
    def strict_e2_boot(
        order_index: int, order: tuple[str, ...], arm: str, order_position: int
    ) -> dict[str, Any]:
        arm_factor = {"S": 1.0, "B2": 0.9, "E2": 0.8}[arm]
        # The first five requests are intentionally not steady-state inputs.
        sample_factors = (1.8, 1.6, 1.4, 1.3, 1.2, 1.0, 1.02, 0.99)
        boot_id = f"boot-{order_index:02d}-{arm}"
        samples = []
        for request_index, factor in enumerate(sample_factors):
            sample = speed_measurement(
                arm,
                wall_seconds=arm_factor * factor,
                ttft_seconds=0.1 * arm_factor * factor,
                predicted_ms=900.0 * arm_factor * factor,
                prompt_ms=100.0 * arm_factor * factor,
            )
            sample["boot_id"] = boot_id
            sample["request_id"] = f"measured-{request_index:02d}"
            samples.append(sample)
        return {
            "boot_id": boot_id,
            "arm": arm,
            "selected_order": "".join(order),
            "order_index": order_index,
            "selected_order_position": order_position,
            "measurements": samples,
        }

    strict_e2_boots = [
        strict_e2_boot(order_index, order, arm, order_position)
        for order_index, order in enumerate(E2_BALANCED_ORDERS)
        for order_position, arm in enumerate(order)
    ]
    strict_e2_measurements = all_measurements(strict_e2_boots)
    assert len(strict_e2_measurements) == 18 * STRICT_E2_MEASURED_PER_BOOT
    assert assert_exact_output_equivalence(strict_e2_measurements) == digest
    e2_summary = speed_summary(
        strict_e2_measurements,
        digest,
        E2_ARMS,
        None,
        mode="balanced",
        expected_orders=E2_BALANCED_ORDERS,
        observed_orders=E2_BALANCED_ORDERS,
        measured_per_boot=STRICT_E2_MEASURED_PER_BOOT,
        request_config=request_config,
        execution_profile="fast-default",
        boots=strict_e2_boots,
    )
    assert set(e2_summary["arms"]) == {"S", "B2", "E2"}
    assert e2_summary["schedule"]["label"] == "complete frozen six-order counterbalanced S/B2/E2 schedule"
    assert e2_summary["comparability"]["cross_arm_speed_scope"] == (
        "same-receipt complete counterbalanced S/B2/E2 timing comparison"
    )
    assert e2_summary["output_arm_coverage"]["complete_for_required_arms"]
    assert e2_summary["steady_state_convergence"]["all_converged"]
    assert len(e2_summary["steady_state_convergence"]["per_boot"]) == 18
    assert len(e2_summary["steady_state_convergence"]["paired_by_order"]) == 6
    first_boot_metrics = e2_summary["steady_state_convergence"]["per_boot"][0][
        "steady_state"
    ]["metrics"]
    assert first_boot_metrics["response_predicted_ms"]["median"] == 900.0
    assert first_boot_metrics["wall_seconds"]["spread_percent_of_median"] < 5.0
    assert math.isclose(
        e2_summary["steady_state_convergence"]["paired_aggregate"]["E2_vs_B2"][
            "metrics"
        ]["wall_seconds"]["median"],
        0.8 / 0.9,
    )
    e2_ratios = e2_summary["arm_metric_ratios"]["B2_vs_E2"]
    assert e2_ratios["numerator_arm"] == "B2"
    assert e2_ratios["denominator_arm"] == "E2"
    assert math.isclose(
        e2_ratios["metrics"]["wall_tokens_per_second"], 0.8 / 0.9
    )
    assert math.isclose(
        e2_ratios["metrics"]["wall_seconds_per_completion_token"], 0.9 / 0.8
    )
    assert math.isclose(
        e2_ratios["metrics"]["metrics_ttft_seconds_mean"], 0.9 / 0.8
    )
    assert math.isclose(
        e2_ratios["metrics"]["response_predicted_ms_mean"], 0.9 / 0.8
    )
    assert math.isclose(
        e2_summary["arm_metric_ratios"]["S_vs_E2"]["metrics"][
            "wall_tokens_per_second"
        ],
        0.8,
    )
    unstable_e2_boots = json.loads(json.dumps(strict_e2_boots))
    unstable_e2_boots[0]["measurements"][-1]["wall_seconds"] = 2.0
    unstable_summary = strict_e2_convergence_summary(
        unstable_e2_boots, E2_BALANCED_ORDERS
    )
    assert not unstable_summary["all_converged"]
    assert not unstable_summary["per_boot"][0]["steady_state"]["metrics"][
        "wall_seconds"
    ]["converged"]

    def settled_row(request_id: str, active: float, cache: float, footprint: float) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "succeeded": request_id.startswith("warmup-"),
            "output_contract": None if request_id.startswith("warmup-") else contract,
            "lifecycle_settlement": {
                "settled": True,
                "second_settled_state": {"ready": True},
                "memory": {
                    "active_plus_cache_bytes": active + cache,
                    "footprint_bytes": footprint,
                },
            },
        }

    lifecycle_boots = []
    for arm in M1_LIFECYCLE_ARMS:
        lifecycle_boots.append(
            {
                "boot_id": f"lifecycle-{arm}",
                "arm": arm,
                "warmups": [settled_row("warmup-00", 1000.0, 10.0, 900.0)],
                "measurements": [
                    settled_row(f"measured-{index:02d}", 1000.0 + index, 10.0, 900.0 + index)
                    for index in range(M1_LIFECYCLE_MEASURED_PER_BOOT)
                ],
            }
        )
    lifecycle_verdict = m1_lifecycle_verdict(lifecycle_boots)
    assert lifecycle_verdict["passed"]
    unstable_lifecycle_boots = json.loads(json.dumps(lifecycle_boots))
    unstable_lifecycle_boots[0]["measurements"][-1]["lifecycle_settlement"]["memory"][
        "active_plus_cache_bytes"
    ] += M1_LIFECYCLE_MEMORY_TOLERANCE_BYTES + 1.0
    assert not m1_lifecycle_verdict(unstable_lifecycle_boots)["passed"]

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
        "runtime_loader": runtime_loader_provenance(None),
        "m1_lifecycle_diagnostic": m1_lifecycle_diagnostic_provenance(False),
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
        mismatched_manifest = json.loads(json.dumps(current_manifest))
        mismatched_manifest["runtime_loader"]["mode"] = "pinned"
        must_fail(
            lambda: compatible_compare_receipt(receipt, mismatched_manifest),
            "comparison runtime loader mismatch",
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
