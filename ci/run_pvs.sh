#!/bin/sh

PVS_LICENSE="${1}"

pvs-studio-analyzer credentials ${PVS_LICENSE}
pvs-studio-analyzer analyze -f build/compile_commands.json -j && \
plog-converter -t sarif -o pvs-report.sarif PVS-Studio.log && \
cat PVS-Studio.log
