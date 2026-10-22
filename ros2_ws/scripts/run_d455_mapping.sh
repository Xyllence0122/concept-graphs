#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "${script_dir}/.." && pwd)"

set +u
source "${script_dir}/ros_env.sh"
source "${workspace_root}/install/setup.bash"
set -u

export CG_AI_SITE_PACKAGES="${CG_AI_SITE_PACKAGES:-${workspace_root}/.venv/lib/python3.12/site-packages}"
export CG_MODEL_PATH="${CG_MODEL_PATH:-${HOME}/concept_graph_models/yolo26n-seg.pt}"

exec ros2 launch cg_object_segmentation d455_mapping.launch.py "$@"
