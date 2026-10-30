#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"
# Host Python is used only for settings validation and Docker mount arguments.
# All application dependencies are installed inside the image, without a venv.
exec python3 -m backend.startup docker "$@"
