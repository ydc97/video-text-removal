#!/usr/bin/env bash
# One-time environment setup for the video-text-removal skill.
# - downloads model weights to a persistent cache (~380 MB)
# - installs Python dependencies
# Run from anywhere:  bash setup.sh
set -uo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${VTR_WEIGHTS_DIR:-$HOME/.cache/video-text-removal/weights}"
PY=python3

fetch() { # url dest
  echo "downloading $(basename "$2")"
  curl -fL --retry 8 --retry-delay 2 -C - -o "$2" "$1"
}

mkdir -p "$WEIGHTS_DIR"

# ---------- weights ----------
[ -f "$WEIGHTS_DIR/lama.pt" ] || fetch \
  "https://huggingface.co/JosephCatrambone/big-lama-torchscript/resolve/main/lama.pt" \
  "$WEIGHTS_DIR/lama.pt"
[ -f "$WEIGHTS_DIR/ProPainter.pth" ] || fetch \
  "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth" \
  "$WEIGHTS_DIR/ProPainter.pth"
[ -f "$WEIGHTS_DIR/raft-things.pth" ] || fetch \
  "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth" \
  "$WEIGHTS_DIR/raft-things.pth"
[ -f "$WEIGHTS_DIR/recurrent_flow_completion.pth" ] || fetch \
  "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth" \
  "$WEIGHTS_DIR/recurrent_flow_completion.pth"

# ---------- python deps ----------
pip_install() { $PY -m pip install "$@"; }
pip_install_user() { $PY -m pip install --user --break-system-packages "$@"; }

try_install() { # tries a plain install first (venv/conda), falls back to PEP-668 user install
  if ! pip_install "$@" >/dev/null 2>&1; then
    pip_install_user "$@" >/dev/null 2>&1 || { echo "pip install failed: $*"; exit 1; }
  fi
}

if ! $PY -c "import torch, torchvision" 2>/dev/null; then
  echo "installing torch + torchvision (CPU build)"
  try_install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  # torchvision must be built against the installed torch - force the pair from the same index
  $PY -m pip install --user --break-system-packages --force-reinstall --no-deps \
    torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true
fi
try_install opencv-python-headless numpy pillow imageio-ffmpeg einops scipy matplotlib tqdm

# ---------- verify ----------
$PY - <<'PY'
import importlib, os, sys
for mod in ("cv2", "numpy", "torch", "torchvision", "PIL", "einops", "scipy", "imageio_ffmpeg", "tqdm"):
    importlib.import_module(mod)
wd = os.environ.get("VTR_WEIGHTS_DIR", os.path.expanduser("~/.cache/video-text-removal/weights"))
need = {"lama.pt": 150_000_000, "ProPainter.pth": 150_000_000,
        "raft-things.pth": 20_000_000, "recurrent_flow_completion.pth": 19_000_000}
missing = [n for n, s in need.items() if not os.path.isfile(os.path.join(wd, n))
           or os.path.getsize(os.path.join(wd, n)) < s]
if missing:
    print("missing/short weights:", missing); sys.exit(1)
print("setup OK - weights in", wd)
PY
