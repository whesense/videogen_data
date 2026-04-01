#!/usr/bin/env bash
# Thin wrapper: runs a single scenario pipeline.
# Usage: ./run.sh 01_spatad_cycle [--config configs/scene_001.yaml] [--steps render save_pairs]
exec python "$(dirname "$0")/run.py" "$@"
