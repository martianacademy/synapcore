#!/usr/bin/env bash
set -e
LMS_PY="/Users/suru.martian/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac14-arm64@39/bin/python3"
if [ ! -x "$LMS_PY" ]; then
  LMS_PY="$HOME/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac14-arm64@39/bin/python3"
fi

# Verify if Metal GPU device is accessible in the current session (true in macOS Terminal, false in headless sandbox)
if [ -x "$LMS_PY" ] && "$LMS_PY" -c "import mlx.core as mx, numpy; assert mx.metal.is_available()" >/dev/null 2>&1; then
  PYTHON_BIN="$LMS_PY"
elif [ -x "/opt/homebrew/bin/python3" ]; then
  PYTHON_BIN="/opt/homebrew/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================================================="
echo "STEP 1: COMPILING C++17 / OBJECTIVE-C++ METAL CORE & BUILDING merkle_tree.idx"
echo "=============================================================================="
"$PYTHON_BIN" "$SCRIPT_DIR/native/build_native_db.py"

echo "=============================================================================="
echo "STEP 2: LAUNCHING STANDALONE SYNAPCORE NATIVE C++17 / MLX-METAL CHAT"
echo "=============================================================================="
exec "$PYTHON_BIN" "$SCRIPT_DIR/native/native_chat.py" "$@"
