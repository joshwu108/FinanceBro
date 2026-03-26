#!/bin/bash
# Build the C++ QAOA kernels extension module.
# Usage: cd quantum/cpp && bash build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR"

cmake --build . --config Release -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

# Copy the .so/.dylib to the parent directory so Python can import it
cp qaoa_cpp*.so "$OUTPUT_DIR/" 2>/dev/null || true
cp qaoa_cpp*.dylib "$OUTPUT_DIR/" 2>/dev/null || true

echo "Build complete. Extension module:"
ls "$OUTPUT_DIR"/qaoa_cpp* 2>/dev/null | grep -v CMake | grep -v build
