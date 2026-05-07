# source /workspace/chuser shirokov
# conda activate /home/jovyan/.mlspace/envs/neurad

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# cd "${REPO_ROOT}" || exit 1

cd /home/jovyan/users/shirokov/videogen_data/

# Ensure Ray is available in runtime environment.
python -m pip install -U ray
# Add user-level bin path where pip installs ray CLI.
export PATH="$(python -m site --user-base)/bin:/home/user/.local/bin:/home/jovyan/users/shirokov/.local/bin:${PATH}"
# Give workers more time to finish Ray install and join cluster.
export RAY_DRIVER_DELAY="${RAY_DRIVER_DELAY:-300}"

# Run through OpenMPI-aware Ray cluster launcher.
exec "run_jobs/run_ray_cluster.sh" 04_3dgs_dl3dv --max-parallel 24 --scene-csv "/home/jovyan/users/shirokov/videogen_data/scenarios/04_3dgs_dl3dv/configs/scenes.csv" "$@"