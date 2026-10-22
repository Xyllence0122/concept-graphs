#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "${script_dir}/.." && pwd)"
errors=0
warnings=0

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; errors=$((errors + 1)); }

if [ "$(uname -m)" = "x86_64" ]; then
    pass "architecture is x86_64"
else
    fail "tested lock targets x86_64; detected $(uname -m)"
fi

source /etc/os-release
if [ "${ID}" = "ubuntu" ] && [ "${VERSION_ID}" = "24.04" ]; then
    pass "operating system is Ubuntu 24.04"
else
    fail "expected Ubuntu 24.04; detected ${ID} ${VERSION_ID}"
fi

if [ -f /opt/ros/jazzy/setup.bash ]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
    pass "ROS 2 Jazzy setup is present"
else
    fail "/opt/ros/jazzy/setup.bash is missing"
fi

for package in \
    realsense2_camera \
    rtabmap_launch \
    rmw_cyclonedds_cpp \
    cv_bridge; do
    if command -v ros2 >/dev/null 2>&1 && ros2 pkg prefix "${package}" >/dev/null 2>&1; then
        pass "ROS package ${package} is installed"
    else
        fail "ROS package ${package} is missing"
    fi
done

if [ "${ROS_DOMAIN_ID:-}" = "73" ]; then
    pass "ROS_DOMAIN_ID=73"
else
    warn "ROS_DOMAIN_ID is '${ROS_DOMAIN_ID:-unset}', expected 73"
fi

if [ "${RMW_IMPLEMENTATION:-}" = "rmw_cyclonedds_cpp" ]; then
    pass "CycloneDDS RMW is selected"
else
    warn "RMW_IMPLEMENTATION is '${RMW_IMPLEMENTATION:-unset}'"
fi

case "${ROS_AUTOMATIC_DISCOVERY_RANGE:-}" in
    LOCALHOST)
        pass "DDS discovery is LOCALHOST (single-machine mode)"
        ;;
    SUBNET)
        pass "DDS discovery is SUBNET (multi-machine mode)"
        ;;
    *)
        warn "ROS_AUTOMATIC_DISCOVERY_RANGE is '${ROS_AUTOMATIC_DISCOVERY_RANGE:-unset}'"
        ;;
esac

ai_python="${CG_AI_PYTHON:-${workspace_root}/.venv/bin/python}"
if [ -x "${ai_python}" ]; then
    if "${ai_python}" - <<'PY'
import cv2
import numpy
import torch
import ultralytics

print(f"      Python AI stack: torch={torch.__version__}, "
      f"CUDA={torch.version.cuda}, available={torch.cuda.is_available()}, "
      f"ultralytics={ultralytics.__version__}, numpy={numpy.__version__}, "
      f"opencv={cv2.__version__}")
raise SystemExit(0 if numpy.__version__.startswith("1.26.") else 1)
PY
    then
        pass "AI Python imports and NumPy ABI guard passed"
    else
        fail "AI Python import or NumPy 1.26 compatibility check failed"
    fi
else
    fail "AI Python environment not found at ${ai_python}"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    pass "NVIDIA driver responds"
else
    warn "NVIDIA driver unavailable; device:auto will fall back to CPU"
fi

model_path="${CG_MODEL_PATH:-${HOME}/concept_graph_models/yolo26n-seg.pt}"
expected_sha="361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"
if [ -f "${model_path}" ]; then
    actual_sha="$(sha256sum "${model_path}" | awk '{print $1}')"
    if [ "${actual_sha}" = "${expected_sha}" ]; then
        pass "YOLO model checksum matches the tested artifact"
    else
        warn "YOLO model exists but checksum differs: ${actual_sha}"
    fi
else
    fail "YOLO model is missing at ${model_path}"
fi

if command -v rs-enumerate-devices >/dev/null 2>&1 \
    && timeout 5 rs-enumerate-devices -s 2>/dev/null | grep -q 'RealSense D455'; then
    pass "RealSense D455 is visible"
else
    warn "RealSense D455 is not visible (acceptable when replaying a rosbag)"
fi

if [ -f "${workspace_root}/install/setup.bash" ]; then
    pass "colcon workspace has been built"
else
    fail "${workspace_root}/install/setup.bash is missing"
fi

printf '\nSummary: %d error(s), %d warning(s)\n' "${errors}" "${warnings}"
exit "${errors}"
