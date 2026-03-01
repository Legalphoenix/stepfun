#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"

if ! python3 - <<'PY'
import importlib.util
import sys
missing = [m for m in ("gradio", "requests") if importlib.util.find_spec(m) is None]
if importlib.util.find_spec("edge_tts") is None:
    missing.append("edge_tts")
sys.exit(1 if missing else 0)
PY
then
  python3 -m pip install --user --upgrade gradio requests edge-tts
fi

python3 runpod/launcher_ui.py
