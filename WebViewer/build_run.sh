#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from this script so it can be launched from any directory.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"

# Prefer the backend venv, then a project-root venv, otherwise create one.
if [[ -x "$PROJECT_DIR/backend/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/backend/.venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    python3 -m venv "$PROJECT_DIR/backend/.venv"
    PYTHON="$PROJECT_DIR/backend/.venv/bin/python"
fi
# pip also brings an existing venv up to date with the requirements file.
"$PYTHON" -m pip install -r "$PROJECT_DIR/backend/requirements.txt"
exec "$PYTHON" -m backend.launch
