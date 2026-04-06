#!/usr/bin/env bash
# Start Ray on an Open MPI / PMIx multi-node job, then run the pipeline driver on rank 0.
#
# Usage (from repo root, after activating your conda env if needed):
#   ./run_jobs/run_ray_cluster.sh <scenario> [ray_run.py args...]
# Example:
#   ./run_jobs/run_ray_cluster.sh 02_neurad_cycle --max-parallel 0
#
# Environment (typical):
#   OMPI_COMM_WORLD_RANK   — 0 = head + driver; >0 = worker (default: 0 if unset)
#   OMPI_COMM_WORLD_SIZE   — number of MPI ranks / nodes (default: 1)
#   PMIX_HOSTNAME          — used with Perl below if MASTER_ADDR is unset
#   MASTER_ADDR            — head node hostname/IP reachable by all nodes (set explicitly or derived)
#
# Optional:
#   RAY_PORT=6379
#   RAY_WORKER_DELAY=15     — sleep on workers before ray start (head must be up first)
#   RAY_DRIVER_DELAY=5      — sleep on rank 0 after head, before ray_run (lets workers join)
#   RUN_RAY_DRIVER=1        — set 0 to only start Ray cluster (no ray_run.sh on rank 0)
#   RAY_LOCAL_ROOT          — passed implicitly via ray_cluster.sh (local SSD if repo is on NFS)
#
# Master address resolution:
#   1) If MASTER_ADDR is already set, use it.
#   2) Else if PMIX_HOSTNAME is set, derive master FQDN like self-driving (mpimaster-0 + prefix).
#   3) Else fail with a clear message.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <scenario> [ray_run.py arguments...]" >&2
  echo "Example: $0 02_neurad_cycle --ray-address auto --max-parallel 0" >&2
  exit 1
fi

RAY_PORT="${RAY_PORT:-6379}"
RANK="${OMPI_COMM_WORLD_RANK:-0}"
WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:-1}"
RAY_WORKER_DELAY="${RAY_WORKER_DELAY:-15}"
RAY_DRIVER_DELAY="${RAY_DRIVER_DELAY:-5}"
RUN_RAY_DRIVER="${RUN_RAY_DRIVER:-1}"

resolve_master_addr() {
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    echo "${MASTER_ADDR}"
    return 0
  fi
  if [[ -z "${PMIX_HOSTNAME:-}" ]]; then
    echo "ERROR: Set MASTER_ADDR to the Ray head's hostname or IP, or run under Open MPI with PMIX_HOSTNAME set." >&2
    exit 1
  fi
  local prefix host
  prefix="$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+\$//; print \$x")"
  host="$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+\$/-mpimaster-0/; print \$x")"
  echo "${host}.${prefix}"
}

MASTER_ADDR="$(resolve_master_addr)"
export RAY_HEAD_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"

echo "run_ray_cluster: REPO_ROOT=${REPO_ROOT}"
echo "run_ray_cluster: RANK=${RANK} WORLD_SIZE=${WORLD_SIZE} MASTER_ADDR=${MASTER_ADDR} RAY_PORT=${RAY_PORT}"

if [[ "${RANK}" -eq 0 ]]; then
  ./ray_cluster.sh head
  if [[ "${WORLD_SIZE}" -gt 1 ]]; then
    echo "run_ray_cluster: waiting ${RAY_DRIVER_DELAY}s for workers to join ..."
    sleep "${RAY_DRIVER_DELAY}"
  fi
else
  echo "run_ray_cluster: worker waiting ${RAY_WORKER_DELAY}s before connecting to head ..."
  sleep "${RAY_WORKER_DELAY}"
  ./ray_cluster.sh worker
fi

if [[ "${RANK}" -eq 0 ]] && [[ "${RUN_RAY_DRIVER}" == "1" ]]; then
  export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
  exec ./ray_run.sh "$@"
fi

exit 0
