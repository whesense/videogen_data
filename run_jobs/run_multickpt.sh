source /workspace/chuser shirokov
conda activate /home/jovyan/.mlspace/envs/neurad

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

DEFAULT_CONFIG="scenarios/03_spatad_cycle_multickpt/configs/scene_0ab21841.yaml"
exec python run.py 03_spatad_cycle_multickpt --config "${MULTICKPT_CONFIG:-$DEFAULT_CONFIG}" "$@"
