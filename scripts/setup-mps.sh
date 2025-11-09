#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script targets macOS hosts; skip on other platforms." >&2
    exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
    cat <<'EOF' >&2
Xcode Command Line Tools are required. Install them with:
  xcode-select --install
EOF
    exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    cat <<'EOF' >&2
Homebrew is required to install the remaining dependencies. Install it via:
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
EOF
    exit 1
fi

deps=(cmake ninja doxygen yaml-cpp)

echo "Ensuring Homebrew packages are installed: ${deps[*]}"

for dep in "${deps[@]}"; do
    if ! brew list --formula "${dep}" >/dev/null 2>&1; then
        echo "Installing ${dep}…"
        brew install "${dep}"
    else
        echo "Already installed: ${dep}"
    fi
done

echo "macOS setup complete. You can now run:"
echo "  cmake -S . -B build -DORTEAF_FETCH_GTEST=ON"
