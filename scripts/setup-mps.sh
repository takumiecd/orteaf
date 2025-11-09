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

if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "xcodebuild is required; install full Xcode from the App Store." >&2
    exit 1
fi

XCODE_DEV_DIR="/Applications/Xcode.app/Contents/Developer"
if [[ ! -d "${XCODE_DEV_DIR}" ]]; then
    echo "Unable to locate ${XCODE_DEV_DIR}. Please install full Xcode." >&2
    exit 1
fi

CURRENT_DEV_DIR=$(xcode-select -p)
if [[ "${CURRENT_DEV_DIR}" != "${XCODE_DEV_DIR}" ]]; then
    echo "Switching active developer directory to ${XCODE_DEV_DIR}"
    sudo xcode-select -s "${XCODE_DEV_DIR}"
fi

echo "Verifying Metal toolchain..."
if ! TOOLCHAINS=metal xcrun -sdk macosx metal -help >/dev/null 2>&1; then
    cat <<'EOF'
Metal command-line tools are missing. Downloading and installing the Metal Toolchain
component (sudo password required).
EOF
    sudo xcodebuild -downloadComponent MetalToolchain

    DMG=$(ls -t /System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain/*/AssetData/Restore/*.dmg 2>/dev/null | head -n1 || true)
    if [[ -z "${DMG}" ]]; then
        echo "Unable to locate downloaded Metal Toolchain DMG." >&2
        exit 1
    fi

    MOUNT=$(mktemp -d)
    cleanup() {
        if mount | grep -q "${MOUNT}"; then
            sudo hdiutil detach "${MOUNT}" >/dev/null 2>&1 || true
        fi
        rmdir "${MOUNT}" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT

    sudo hdiutil attach "${DMG}" -mountpoint "${MOUNT}" -nobrowse >/dev/null

    if [[ -f "${MOUNT}/Metal Tools.pkg" ]]; then
        sudo installer -pkg "${MOUNT}/Metal Tools.pkg" -target /
    elif [[ -d "${MOUNT}/Metal.xctoolchain" ]]; then
        TARGET_DIR="/Applications/Xcode.app/Contents/Developer/Toolchains"
        sudo rm -rf "${TARGET_DIR}/Metal.xctoolchain"
        sudo cp -R "${MOUNT}/Metal.xctoolchain" "${TARGET_DIR}"
    else
        echo "Unsupported Metal Toolchain payload: ${MOUNT}" >&2
        exit 1
    fi

    cleanup
    trap - EXIT

    if ! TOOLCHAINS=metal xcrun -sdk macosx metal -help >/dev/null 2>&1; then
        echo "Metal toolchain still unavailable after installation attempt." >&2
        exit 1
    fi
else
    echo "Metal toolchain already present."
fi

echo "macOS setup complete. You can now run:"
echo "  cmake -S . -B build -DORTEAF_FETCH_GTEST=ON"
