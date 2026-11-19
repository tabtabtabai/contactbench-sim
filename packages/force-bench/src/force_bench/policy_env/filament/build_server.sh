#!/usr/bin/env bash
# Build filament_server against a Filament release (static libs; clang + libc++ like the release itself).
# usage: build_server.sh FILAMENT_DIR OUT   e.g. build_server.sh ~/fil/filament ~/fl/filament_server
# Needs: clang, libc++-dev, libc++abi-dev, libzstd-dev, libx11-dev, and the NVIDIA Vulkan ICD for running it.
set -euo pipefail
F=$1; OUT=$2; L=$F/lib/x86_64
clang++ -std=c++20 -stdlib=libc++ -O2 -I"$F/include" "$(dirname "$0")/server.cpp" -o "$OUT" -L"$L" \
  -Wl,--start-group -lgltfio_core -luberarchive -lktxreader -limage -lfilament -lbackend -lbluevk -lbluegl \
  -lfilabridge -lfilaflat -lutils -lgeometry -lsmol-v -libl-lite -ldracodec -lmeshoptimizer -lstb -luberzlib \
  -lmikktspace -labseil -lperfetto -lshaders -lwebgpu_dawn -Wl,--end-group -lzstd -lX11 -lc++ -lc++abi -lpthread -ldl -lm
echo "built $OUT"
