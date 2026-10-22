#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "${script_dir}/.." && pwd)"

set +u
source /opt/ros/jazzy/setup.bash
set -u

cd "${workspace_root}"
colcon build --symlink-install \
    --packages-select cg_depth_projection cg_object_segmentation
colcon test \
    --packages-select cg_depth_projection cg_object_segmentation \
    --event-handlers console_direct+
colcon test-result --verbose
