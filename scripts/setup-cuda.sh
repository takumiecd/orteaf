#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname)" != "Linux" ]]; then
    cat <<'EOF' >&2
CUDA setup currently targets Linux hosts. For macOS Metal, use scripts/setup-mps.sh instead.
EOF
    exit 1
fi

echo "Running baseline CPU toolchain setup before CUDA configuration..."
"${SCRIPT_DIR}/setup-cpu.sh"

require_command() {
    local name="$1"
    local install_hint="$2"
    if ! command -v "${name}" >/dev/null 2>&1; then
        cat <<EOF >&2
Required command '${name}' not found.
${install_hint}
EOF
        exit 1
    fi
}

require_command nvidia-smi "Install NVIDIA drivers and ensure the NVIDIA kernel modules are loaded."
require_command nvcc "Install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads or your distribution packages."
require_command llvm-objcopy "Install LLVM (apt-get install llvm) so that llvm-objcopy is available."

if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    cat <<'EOF' >&2
CUDA_HOME is not set and /usr/local/cuda is missing. Install the CUDA Toolkit and set CUDA_HOME accordingly.
EOF
    exit 1
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    cat <<EOF >&2
nvcc was found in PATH but not under ${CUDA_HOME}.
Please ensure CUDA_HOME (${CUDA_HOME}) matches the toolkit providing nvcc.
EOF
    exit 1
fi

cat <<'EOF'
CUDA setup complete. Typical workflow inside this environment:
  cmake -S . -B build -DENABLE_CUDA=ON -DENABLE_CPU=ON -DENABLE_MPS=OFF
  cmake --build build -j
  cd build && ctest --output-on-failure

When running inside Docker, use docker/run-cuda.sh to start a container with --gpus=all.
EOF
