#!/usr/bin/env bash
# A safetensors index selects the active checkpoint for residency accounting.
# Extra root weight files must not inflate /v1/models or Ollama tag sizes.
set -uo pipefail

PORT="${1:-11474}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlxserve-index-bytes.XXXXXX")"
SRV=""
trap 'rm -rf "$TMP"; [ -n "$SRV" ] && kill "$SRV" 2>/dev/null' EXIT

MODEL="$TMP/models/org/indexed-model"
mkdir -p "$MODEL"
printf '{"model_type":"qwen3"}' > "$MODEL/config.json"
printf '{"weight_map":{"a.weight":"model.safetensors","b.weight":"model.safetensors"}}' \
  > "$MODEL/model.safetensors.index.json"
printf '0123456789' > "$MODEL/model.safetensors"
printf 'obsolete' > "$MODEL/model-00001-of-00002.safetensors"
printf 'unused' > "$MODEL/model-00002-of-00002.safetensors"

"$BIN" --serve --port "$PORT" --model-dir "$TMP/models" > "$TMP/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server did not start"; tail -10 "$TMP/server.log"; exit 1; }
  sleep 0.5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "FAIL: server never became healthy"; exit 1; }

V1_BYTES="$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['data'][0]['bytes_on_disk'])")"
TAG_BYTES="$(curl -sf "http://127.0.0.1:$PORT/api/tags" | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['models'][0]['size'])")"

if [ "$V1_BYTES" = "10" ] && [ "$TAG_BYTES" = "10" ]; then
  echo "PASS: indexed checkpoint reports 10 bytes on both model APIs"
else
  echo "FAIL: expected 10 indexed bytes, got v1=$V1_BYTES tags=$TAG_BYTES"
  exit 1
fi
