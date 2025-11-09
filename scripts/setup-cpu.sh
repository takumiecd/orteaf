#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ensure_yaml_cpp_linux() {
    local required_version="0.8.0"
    local have_version=""
    if pkg-config --exists yaml-cpp 2>/dev/null; then
        have_version="$(pkg-config --modversion yaml-cpp 2>/dev/null || true)"
    fi

    version_ge() {
        local IFS=.
        read -r -a lhs <<<"$1"
        read -r -a rhs <<<"$2"
        local len="${#lhs[@]}"
        if [[ "${#rhs[@]}" -gt "${len}" ]]; then
            len="${#rhs[@]}"
        fi
        for ((i=0; i<len; ++i)); do
            local l="${lhs[i]:-0}"
            local r="${rhs[i]:-0}"
            if ((10#${l} > 10#${r})); then
                return 0
            elif ((10#${l} < 10#${r})); then
                return 1
            fi
        done
        return 0
    }

    if [[ -n "${have_version}" ]] && version_ge "${have_version}" "${required_version}"; then
        echo "yaml-cpp ${have_version} already available (>= ${required_version})."
        return
    fi

    echo "Building yaml-cpp ${required_version} from source (requires sudo)."
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    git clone --depth 1 --branch "${required_version}" https://github.com/jbeder/yaml-cpp.git "${tmp_dir}/yaml-cpp"
    cmake -S "${tmp_dir}/yaml-cpp" -B "${tmp_dir}/build" \
        -DYAML_BUILD_SHARED_LIBS=ON \
        -DYAML_CPP_BUILD_TOOLS=OFF
    cmake --build "${tmp_dir}/build" -j"$(nproc)"
    sudo cmake --install "${tmp_dir}/build"
    rm -rf "${tmp_dir}"
    sudo ldconfig
}

case "$(uname)" in
    Darwin)
        if ! command -v brew >/dev/null 2>&1; then
            cat <<'EOF' >&2
Homebrew is required to install dependencies. Install it via:
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
EOF
            exit 1
        fi

        brew_deps=(cmake ninja doxygen graphviz yaml-cpp llvm)
        echo "Ensuring Homebrew packages are installed: ${brew_deps[*]}"
        for dep in "${brew_deps[@]}"; do
            if ! brew list --formula "${dep}" >/dev/null 2>&1; then
                echo "Installing ${dep}..."
                brew install "${dep}"
            else
                echo "Already installed: ${dep}"
            fi
        done
        ;;
    Linux)
        if ! command -v apt-get >/dev/null 2>&1; then
            cat <<'EOF' >&2
This script currently supports Debian/Ubuntu via apt-get.
Please install the following packages manually and re-run the build:
  build-essential clang llvm cmake ninja-build git python3 python3-pip doxygen graphviz ccache pkg-config curl
EOF
            exit 1
        fi

        echo "Installing base toolchain packages via apt-get (sudo required)."
        sudo apt-get update
        sudo apt-get install -y --no-install-recommends \
            build-essential \
            clang \
            llvm \
            cmake \
            ninja-build \
            git \
            python3 \
            python3-pip \
            doxygen \
            graphviz \
            ccache \
            pkg-config \
            curl

        if ! command -v git >/dev/null 2>&1; then
            echo "git installation failed; please install manually." >&2
            exit 1
        fi
        ensure_yaml_cpp_linux
        ;;
    *)
        cat <<'EOF' >&2
Unsupported host platform. Please install cmake, ninja, clang/llvm, doxygen, graphviz, pkg-config,
and yaml-cpp >= 0.8.0 manually, then follow the build instructions in README.md.
EOF
        exit 1
        ;;
esac

cat <<'EOF'
CPU toolchain setup complete. Typical build sequence:
  cmake -S . -B build -DENABLE_CPU=ON
  cmake --build build
  cd build && ctest --output-on-failure
EOF
