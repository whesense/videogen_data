#!/bin/sh

source /workspace/chuser shirokov
conda activate  /home/jovyan/.mlspace/envs/neurad
cd airi/videogen_data/
./ray_run.sh 01_spatad_cycle
