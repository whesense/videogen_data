#!/usr/bin/env bash
# Start/stop a multi-node Ray cluster (e.g. 2 machines: one head + one worker).
#
# Head (node A):
#   ./ray_cluster.sh head
#   Optional: RAY_PORT=6379 NUM_GPUS=1 ./ray_cluster.sh head
#
# Worker (node B) — use the head’s reachable IP and the same port:
#   RAY_HEAD_ADDRESS=<head_ip>:6379 ./ray_cluster.sh worker
#
# Driver (usually on the head, after both are up):
#   export RAY_ADDRESS=<head_ip>:6379
#   ./ray_run.sh 01_spatad_cycle --ray-address auto --max-parallel 0
#
# Stop Ray on a host:
#   ./ray_cluster.sh stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAY_LOCAL_ROOT="${RAY_LOCAL_ROOT:-$SCRIPT_DIR/.ray_local}"
mkdir -p "$RAY_LOCAL_ROOT/tmp" "$RAY_LOCAL_ROOT/spill"
# Workers ignore --temp-dir; TMPDIR/RAY_TMPDIR steers Ray temp files to the same root.
export TMPDIR="${TMPDIR:-$RAY_LOCAL_ROOT/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-$RAY_LOCAL_ROOT/tmp}"

RAY_PORT="${RAY_PORT:-6379}"
ROLE="${1:-}"
[[ -n "$ROLE" ]] && shift

case "$ROLE" in
  head)
    exec ray start --head \
      --port="${RAY_PORT}" \
      --dashboard-host=0.0.0.0 \
      --temp-dir="${RAY_LOCAL_ROOT}/tmp" \
      --object-spilling-directory="${RAY_LOCAL_ROOT}/spill" \
      ${NUM_GPUS:+--num-gpus "${NUM_GPUS}"} \
      "$@"
    ;;
  worker)
    : "${RAY_HEAD_ADDRESS:?Set RAY_HEAD_ADDRESS to head address, e.g. 10.0.0.1:6379}"
    exec ray start --address="${RAY_HEAD_ADDRESS}" \
      --object-spilling-directory="${RAY_LOCAL_ROOT}/spill" \
      ${NUM_GPUS:+--num-gpus "${NUM_GPUS}"} \
      "$@"
    ;;
  stop)
    exec ray stop "$@"
    ;;
  *)
    echo "Usage: $0 head|worker|stop [extra args passed to ray start/stop]" >&2
    echo "See comments in $0 for multi-node setup." >&2
    exit 1
    ;;
esac
