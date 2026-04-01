#!/usr/bin/env bash
# Thin wrapper: distributed run — 1 Ray job per config.
# Usage: ./ray_run.sh 01_spatad_cycle [--ray-address auto] [--max-parallel N] [--num-gpus 1] [--dry-run]  (default max-parallel=8)
exec python "$(dirname "$0")/ray_run.py" "$@"
