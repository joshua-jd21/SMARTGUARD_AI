#!/usr/bin/env bash
# Create .venv for SmartGuard AI on macOS.
# 1) Prefer arm64 Homebrew Python 3.10 + full requirements.txt (TensorFlow).
# 2) If only Rosetta/x86_64 Python on Apple Silicon: use requirements-agents-ui.txt (no TensorFlow).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

is_apple_silicon() {
  local b
  b="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  [[ "$b" == *Apple* ]]
}

pick_arm64_py310() {
  local p arch candidates bp
  candidates=(
    "/opt/homebrew/opt/python@3.10/bin/python3.10"
    "/opt/homebrew/bin/python3.10"
  )
  if command -v brew >/dev/null 2>&1; then
    bp="$(brew --prefix python@3.10 2>/dev/null || true)"
    if [[ -n "${bp:-}" && -x "$bp/bin/python3.10" ]]; then
      candidates+=("$bp/bin/python3.10")
    fi
  fi
  for p in "${candidates[@]}"; do
    [[ -x "$p" ]] || continue
    arch="$( "$p" -c "import platform; print(platform.machine())" 2>/dev/null || true )"
    if [[ "$arch" == "arm64" ]]; then
      ver="$( "$p" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" )"
      if [[ "$ver" == "3.10" ]]; then
        echo "$p"
        return 0
      fi
    fi
  done
  return 1
}

pick_any_py310() {
  local p ver
  for p in $(command -v python3.10 2>/dev/null); do
    [[ -x "$p" ]] || continue
    ver="$( "$p" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || true )"
    if [[ "$ver" == "3.10" ]]; then
      echo "$p"
      return 0
    fi
  done
  if command -v which >/dev/null 2>&1; then
    for p in $(which -a python3.10 2>/dev/null); do
      [[ -x "$p" ]] || continue
      ver="$( "$p" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || true )"
      if [[ "$ver" == "3.10" ]]; then
        echo "$p"
        return 0
      fi
    done
  fi
  return 1
}

REQ="requirements.txt"
NO_TF_MARKER=0

if PY="$(pick_arm64_py310)"; then
  echo "Using arm64 Python 3.10: $PY"
elif is_apple_silicon && PY="$(pick_any_py310)"; then
  arch="$( "$PY" -c "import platform; print(platform.machine())" )"
  if [[ "$arch" == "arm64" ]]; then
    echo "Using arm64 Python 3.10: $PY"
    REQ="requirements.txt"
  else
    echo "No /opt/homebrew arm64 Python 3.10 found (install: https://brew.sh then: brew install python@3.10)."
    echo "Falling back to UI-only stack (agents + Streamlit + MQTT; no TensorFlow): $PY ($arch)"
    REQ="requirements-agents-ui.txt"
    NO_TF_MARKER=1
  fi
else
  echo "No usable Python 3.10 found."
  echo "Install Apple Silicon Homebrew to /opt/homebrew, then:  brew install python@3.10"
  echo "Re-run:  bash scripts/bootstrap_macos_arm64_venv.sh"
  exit 1
fi

rm -rf .venv
"$PY" -m venv .venv
./.venv/bin/python -m pip install -U pip setuptools wheel
./.venv/bin/python -m pip install -r "$REQ"

if [[ "$NO_TF_MARKER" -eq 1 ]]; then
  touch .venv/SMARTGUARD_NO_TF
fi

echo ""
echo "Done ($REQ)."
echo "  source .venv/bin/activate"
echo "  python check_setup.py"
if [[ "$NO_TF_MARKER" -eq 1 ]]; then
  echo "  (Deep learning: install arm64 Python + pip install -r requirements.txt, then remove .venv/SMARTGUARD_NO_TF)"
fi
