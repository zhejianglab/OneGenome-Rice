#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/default_config.json}"

CONDA_SH="$(python - "${CONFIG}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    cfg = json.load(fh)
print(cfg["environment"].get("conda_sh", ""))
PY
)"
CONDA_ENV="${CONDA_ENV:-$(python - "${CONFIG}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    cfg = json.load(fh)
print(cfg["environment"].get("conda_env", ""))
PY
)}"

if [[ -n "${CONDA_ENV}" ]]; then
    if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
        source "${CONDA_SH}"
    fi
    conda run --no-capture-output -n "${CONDA_ENV}" \
        python "${ROOT_DIR}/Scripts/env_check.py" --config "${CONFIG}" "$@"
else
    python "${ROOT_DIR}/Scripts/env_check.py" --config "${CONFIG}" "$@"
fi
