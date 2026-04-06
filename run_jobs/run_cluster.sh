#!/usr/bin/env bash
source /workspace/chuser shirokov
conda activate  /home/jovyan/.mlspace/envs/neurad

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

exec "${SCRIPT_DIR}/run_ray_cluster.sh" 01_spatad_cycle "$@"
