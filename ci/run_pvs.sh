#!/bin/sh

PVS_LICENSE="${1}"
BUILD_DIR="${2:-build}"

pvs-studio-analyzer credentials ${PVS_LICENSE}
cmake --build "${BUILD_DIR}" --parallel --target pvs-analyze
cat PVS-Studio.log
