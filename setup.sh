#!/usr/bin/env bash

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

if [ -z "${CONDA_PREFIX}" ]; then
    echo "[ERROR] Isaac Lab conda 환경을 먼저 활성화하세요." >&2
    exit 1
fi

source "${REPO_DIR}/env.sh"

python - <<'PY'
import importlib.util
import sys

required = ["isaaclab", "isaaclab_tasks", "gymnasium"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"[ERROR] 누락된 모듈: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)
PY

python -m pip install -e .

echo "[OK] Agibot Toy2Box 프로젝트를 editable mode로 설치했습니다."
echo "[INFO] 새 터미널에서는 'source env.sh'로 Isaac Sim 경로를 설정하세요."
echo "[INFO] BC 학습에는 robomimic v0.4.0이 추가로 필요합니다."
