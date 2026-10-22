#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash
exec "$@"
