#!/bin/zsh
set -euo pipefail

script_dir=${0:A:h}
repo_dir=${script_dir:h}
server_bin=${repo_dir}/zig-out/bin/mlx-serve
model_dir=/Users/pjb/models/mlx-community/DeepSeek-V4-Flash-0731-target-only-gold-view-20260811-v1

if [[ ! -x ${server_bin} ]]; then
  print -u2 "missing ReleaseFast server: ${server_bin}"
  print -u2 "build it with: cd ${repo_dir} && ./.zig-toolchain/zig build -Doptimize=ReleaseFast"
  exit 1
fi

if [[ ! -d ${model_dir} ]]; then
  print -u2 "missing validated model: ${model_dir}"
  exit 1
fi

# The `fit` wired-residency policy is the measured-stable serving policy for
# this ~100 GB working set (28.3 decode tok/s streaming over HTTP, no swap;
# `off` thrash-tested at 0.46 tok/s). Allow a caller override via the same
# env the runtime reads.
: ${MLX_SERVE_WIRED:=fit}
: ${MLX_SERVE_WIRED_SLACK_MB:=0}
export MLX_SERVE_WIRED MLX_SERVE_WIRED_SLACK_MB

exec ${server_bin} \
  --model ${model_dir} \
  --no-pld \
  --no-decode-attn-quant \
  --no-vision \
  --ctx-size ${MLX_SERVE_DSV4_USER_CTX_SIZE:-8192} \
  --timeout ${MLX_SERVE_DSV4_USER_TIMEOUT:-300} \
  "$@"
