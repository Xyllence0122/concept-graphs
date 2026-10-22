#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "${script_dir}/.." && pwd)"
environment_dir="${workspace_root}/environment"

architecture="$(uname -m)"
if [ "${architecture}" != "x86_64" ]; then
    echo "ERROR: this lock set targets x86_64, detected ${architecture}." >&2
    echo "ARM64/Jetson requires a target-specific PyTorch installation." >&2
    exit 2
fi

source /etc/os-release
if [ "${ID}" != "ubuntu" ] || [ "${VERSION_ID}" != "24.04" ]; then
    echo "ERROR: Ubuntu 24.04 is required; detected ${ID} ${VERSION_ID}." >&2
    exit 2
fi

mapfile -t apt_packages < <(
    sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' \
        "${environment_dir}/apt-packages.txt"
)
sudo apt-get update
sudo apt-get install -y "${apt_packages[@]}"

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update --rosdistro jazzy

python3 -m venv --system-site-packages "${workspace_root}/.venv"
ai_python="${workspace_root}/.venv/bin/python"
"${ai_python}" -m pip install \
    pip==26.2.1 setuptools==78.1.0 wheel==0.48.0

torch_flavor="${CG_TORCH_FLAVOR:-cu128}"
case "${torch_flavor}" in
    cu128)
        torch_requirements="${environment_dir}/requirements-torch-cu128.txt"
        ;;
    cpu)
        torch_requirements="${environment_dir}/requirements-torch-cpu.txt"
        ;;
    *)
        echo "ERROR: CG_TORCH_FLAVOR must be cu128 or cpu." >&2
        exit 2
        ;;
esac

"${ai_python}" -m pip install -r "${torch_requirements}"
"${ai_python}" -m pip install -r "${environment_dir}/requirements-ai.txt"

set +u
source /opt/ros/jazzy/setup.bash
set -u
rosdep install \
    --from-paths "${workspace_root}/src" \
    --ignore-src \
    --rosdistro jazzy \
    -r -y

"${script_dir}/build_and_test.sh"

cat <<EOF

Native environment is ready.
Run:
  source "${workspace_root}/scripts/ros_env.sh"
  source "${workspace_root}/install/setup.bash"
  export CG_AI_SITE_PACKAGES="${workspace_root}/.venv/lib/python3.12/site-packages"
  export CG_MODEL_PATH=/absolute/path/to/yolo26n-seg.pt
EOF
