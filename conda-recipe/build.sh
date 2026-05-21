#!/usr/bin/env bash
set -euo pipefail

"${PYTHON}" -m pip install . --no-deps --no-build-isolation -vv

SP_DIR="$("${PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

mkdir -p "${SP_DIR}/trained_models"
cp -R trained_models/. "${SP_DIR}/trained_models/"
