#!/usr/bin/env bash
# Build filament_server (render version 4) reproducibly against a PINNED Filament release.
#
# usage: build_server.sh OUT [WORKDIR]
#   OUT      the binary to write (e.g. ./filament_server)
#   WORKDIR  where the release is downloaded and unpacked (default: a temp dir)
#
# Fetches filament-$FILAMENT_VERSION-linux.tgz from GitHub releases, checks its sha256 against the pin, and links the
# server against its static libraries with clang + libc++ (the release's own toolchain).
# Build deps (Ubuntu 24.04): clang libc++-dev libc++abi-dev libzstd-dev libx11-dev curl
# Runtime: an NVIDIA driver with its Vulkan ICD (no X server; the server renders headless on Vulkan).
set -euo pipefail
FILAMENT_VERSION=v1.81.0
FILAMENT_SHA256=ce4ee046c9ed7254448bca2a08fd25afce5481ec48d8c457391d19bfddda3a1f
OUT=${1:?usage: build_server.sh OUT [WORKDIR]}
WORK=${2:-$(mktemp -d)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK"
TGZ="$WORK/filament-$FILAMENT_VERSION-linux.tgz"
if [ ! -f "$TGZ" ]; then
  curl -sfL -o "$TGZ" "https://github.com/google/filament/releases/download/$FILAMENT_VERSION/filament-$FILAMENT_VERSION-linux.tgz"
fi
echo "$FILAMENT_SHA256  $TGZ" | sha256sum -c - >/dev/null || { echo "filament release checksum mismatch" >&2; exit 1; }
[ -d "$WORK/filament/lib" ] || tar xzf "$TGZ" -C "$WORK"
F=$WORK/filament
L=$F/lib/x86_64
clang++ -std=c++20 -stdlib=libc++ -O2 -I"$F/include" "$HERE/server.cpp" -o "$OUT" -L"$L" \
  -Wl,--start-group -lgltfio_core -luberarchive -lktxreader -limage -lfilament -lbackend -lbluevk -lbluegl \
  -lfilabridge -lfilaflat -lutils -lgeometry -lsmol-v -libl-lite -ldracodec -lmeshoptimizer -lstb -luberzlib \
  -lmikktspace -labseil -lperfetto -lshaders -lwebgpu_dawn -Wl,--end-group -lzstd -lX11 -lc++ -lc++abi -lpthread -ldl -lm
echo "built $OUT against filament $FILAMENT_VERSION"
echo "IBL for TB_FILAMENT_IBL: $F/bin/assets/ibl/lightroom_14b"
