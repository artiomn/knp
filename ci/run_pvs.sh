#!/bin/sh -x

PVS_LICENSE="${1} ${2}"
BUILD_DIR="${3:-build}"

pvs-studio-analyzer credentials ${PVS_LICENSE}
cmake --build "${BUILD_DIR}" --parallel --target pvs-analyze
cat PVS-Studio.log
