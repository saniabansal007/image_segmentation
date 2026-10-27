#!/usr/bin/env bash
# ==============================================================================
# One-Click Launcher for CT Interactive Segmentation Viewer
# ==============================================================================
set -e

# Resolve project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$PROJECT_ROOT"

# Ensure DISPLAY is set for WSL / Linux environments
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    if grep -qi microsoft /proc/version 2>/dev/null; then
        WSL_HOST_IP=$(grep nameserver /etc/resolv.conf 2>/dev/null | awk '{print $2}')
        if [ -n "$WSL_HOST_IP" ]; then
            export DISPLAY="${WSL_HOST_IP}:0"
        else
            export DISPLAY=":0"
        fi
    else
        export DISPLAY=":0"
    fi
fi
export LIBGL_ALWAYS_INDIRECT=0

# Automatically compile ct_viewer if not built yet
if [ ! -f "build/ct_viewer" ]; then
    echo "==> [run.sh] ct_viewer binary not found. Building project..."
    mkdir -p build
    cmake -B build -S .
    cmake --build build --target ct_viewer -j"$(nproc 2>/dev/null || echo 4)"
fi

# Automatically generate preprocessed multi-plane cache if missing
if [ ! -f "data_preprocessed/manifest.json" ] && [ -d "data" ]; then
    echo "==> [run.sh] data_preprocessed/ not detected. Building preprocessor..."
    cmake --build build --target ct_preprocess -j"$(nproc 2>/dev/null || echo 4)"
    echo "==> [run.sh] Preprocessing dataset for instant lazy-loading..."
    ./build/ct_preprocess data -o data_preprocessed
fi

echo "==> [run.sh] Launching CT Interactive Segmentation Viewer..."
exec ./build/ct_viewer "$@"
