#!/bin/sh
# Fast test: short NeuRAD training (override YAML neurad_num_iter). Unset for full runs.
export NEURAD_NUM_ITER="${NEURAD_NUM_ITER:-1000}"

source /workspace/chuser shirokov
conda activate /home/jovyan/.mlspace/envs/neurad
cd /home/jovyan/users/shirokov/airi/videogen_data || exit 1
exec ./ray_run.sh 02_neurad_cycle "$@"
