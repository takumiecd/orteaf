#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to run this script." >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ORTEAF_CUDA_IMAGE:-orteaf-dev-cuda}"

echo "[ORTEAF][CUDA] Building image ${image}"
docker build -f "${repo_root}/docker/cuda/Dockerfile" -t "${image}" "${repo_root}"

if docker info --format '{{range .Runtimes}}{{println .}}{{end}}' >/tmp/orteaf-docker-runtime 2>/dev/null; then
    if ! grep -q nvidia /tmp/orteaf-docker-runtime; then
        cat <<'EOF'
Warning: Docker runtime 'nvidia' not detected. Ensure the NVIDIA Container Toolkit
is installed and --gpus=all works on this host before running CUDA workloads.
EOF
    fi
    rm -f /tmp/orteaf-docker-runtime
else
    cat <<'EOF'
Warning: Unable to query Docker runtimes (docker info failed). CUDA containers may fail
if the NVIDIA runtime is not configured.
EOF
fi

cmd=(/bin/bash)
if [[ $# -gt 0 ]]; then
    cmd=("$@")
fi

echo "[ORTEAF][CUDA] Launching container with GPU access"
docker run --rm -it \
    --gpus=all \
    -v "${repo_root}:/workspace" \
    -w /workspace \
    "${image}" \
    "${cmd[@]}"
