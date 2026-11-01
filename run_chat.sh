#!/usr/bin/env bash
set -e
LMS_PY="$HOME/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac14-arm64@39/bin/python3"
if [ ! -x "$LMS_PY" ]; then
  LMS_PY="$(command -v python3)"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$LMS_PY" "$SCRIPT_DIR/compare_chat.py" "$@"
