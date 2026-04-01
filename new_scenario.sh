#!/usr/bin/env bash
# Create a new scenario from the _template.
#
# Usage:
#   ./new_scenario.sh 02_lidar_noise
#   ./new_scenario.sh 03_weather_augment
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCENARIOS="${REPO_ROOT}/scenarios"
TEMPLATE="${SCENARIOS}/_template"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <scenario_name>"
    echo "Example: $0 02_lidar_noise"
    exit 1
fi

NAME="$1"
DEST="${SCENARIOS}/${NAME}"

if [[ -d "$DEST" ]]; then
    echo "ERROR: Already exists: ${DEST}" >&2
    exit 1
fi

cp -r "$TEMPLATE" "$DEST"

# Add __init__.py for Python import
touch "${DEST}/__init__.py"

# Remove template gitkeeps / example files
rm -f "${DEST}/configs/.gitkeep"
rm -f "${DEST}/scripts/.gitkeep"

echo "Created: ${DEST}"
echo ""
echo "Next steps:"
echo "  1. Edit ${DEST}/base.yaml      — scenario defaults"
echo "  2. Edit ${DEST}/pipeline.py    — implement download/render/save_pairs"
echo "  3. Add per-job configs in ${DEST}/configs/*.yaml"
echo ""
echo "Run single:       python run.py ${NAME}"
echo "Run with config:  python run.py ${NAME} --config scenarios/${NAME}/configs/my_job.yaml"
echo "Run distributed:  python ray_run.py ${NAME}"
