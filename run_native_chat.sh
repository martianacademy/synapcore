#!/usr/bin/env bash
set -e
LMS_PY="$HOME/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac14-arm64@39/bin/python3"
if [ ! -x "$LMS_PY" ]; then
  LMS_PY="$(command -v python3)"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================================================="
echo "STEP 1: COMPILING C++17 / OBJECTIVE-C++ METAL CORE & BUILDING merkle_tree.idx"
echo "=============================================================================="
"$LMS_PY" "$SCRIPT_DIR/native/build_native_db.py"

echo "=============================================================================="
echo "STEP 2: LAUNCHING NATIVE C++17 / METAL SIDE-BY-SIDE CHAT"
echo "=============================================================================="
exec "$LMS_PY" "$SCRIPT_DIR/native/native_chat.py" "$@"
