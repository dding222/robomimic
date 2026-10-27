#!/usr/bin/env bash

ISAACLAB_ROOT="/home/user/package/IsaacLab"

if [ ! -f "${ISAACLAB_ROOT}/_isaac_sim/setup_conda_env.sh" ]; then
    echo "[ERROR] Isaac Sim 환경 설정 파일을 찾을 수 없습니다: ${ISAACLAB_ROOT}/_isaac_sim/setup_conda_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

source "${ISAACLAB_ROOT}/_isaac_sim/setup_conda_env.sh"
export ISAACLAB_PATH="${ISAACLAB_ROOT}"

if [ -n "${CONDA_PREFIX}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
