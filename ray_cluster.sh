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
#
# Dashboard on by default (head): http://<HEAD_IP>:${RAY_DASHBOARD_PORT:-8265} — needs pip install 'ray[default]'.
# Set RAY_INCLUDE_DASHBOARD=0 if dashboard deps are missing. http not https; ERR_EMPTY_RESPONSE → wrong host / firewall / SSH -L.

set -euo pipefail

# Real mount of this script (pwd -P resolves symlinks).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RAY_LOCAL_ROOT="${RAY_LOCAL_ROOT:-$SCRIPT_DIR/.ray_local}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-$RAY_LOCAL_ROOT/spill}"
# Short path for sessions + Unix sockets (see header).
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_${USER:-user}}"

mkdir -p "$RAY_TEMP_DIR" "$RAY_SPILL_DIR"
# Workers ignore --temp-dir; TMPDIR/RAY_TMPDIR must use the same short base.
export TMPDIR="${TMPDIR:-$RAY_TEMP_DIR}"
export RAY_TMPDIR="${RAY_TMPDIR:-$RAY_TEMP_DIR}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-0}"
ROLE="${1:-}"
[[ -n "$ROLE" ]] && shift

_dashboard_args() {
  if [[ "${RAY_INCLUDE_DASHBOARD}" == "1" || "${RAY_INCLUDE_DASHBOARD}" == "true" ]]; then
    echo --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"
  else
    echo --include-dashboard=false
  fi
}

case "$ROLE" in
  head)
    # shellcheck disable=SC2046
    exec ray start --head \
      --port="${RAY_PORT}" \
      $(_dashboard_args) \
      --disable-usage-stats \
      --temp-dir="${RAY_TEMP_DIR}" \
      --object-spilling-directory="${RAY_SPILL_DIR}" \
      ${NUM_GPUS:+--num-gpus "${NUM_GPUS}"} \
      "$@"
    ;;
  worker)
    : "${RAY_HEAD_ADDRESS:?Set RAY_HEAD_ADDRESS to head address, e.g. 10.0.0.1:6379}"
    exec ray start --address="${RAY_HEAD_ADDRESS}" \
      --disable-usage-stats \
      --object-spilling-directory="${RAY_SPILL_DIR}" \
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
