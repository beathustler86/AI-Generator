#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
echo "=== AI Generator Environment Setup ==="

[ -d .venv ] || $PYTHON -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Install numpy & opencv first to avoid temporary numpy 2.x install
pip install numpy==1.26.4 opencv-python==4.10.0.84

pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124

pip install -r requirements.txt -c constraints-gpu.txt --upgrade --upgrade-strategy only-if-needed

# xformers intentionally skipped (native SDPA). Install manually if needed:
# pip install --extra-index-url https://download.pytorch.org/whl/cu124 xformers==<version>

pip install --no-deps "realesrgan @ git+https://github.com/xinntao/Real-ESRGAN.git@a4abfb2979a7bbff3f69f58f58ae324608821e27"

if [ "${DEV:-0}" = "1" ]; then
  pip install -r requirements-dev.txt -c constraints-gpu.txt
fi

python - <<'PY'
import torch, importlib, cv2, numpy, json, platform
print(json.dumps({
  "torch": torch.__version__,
  "cuda": torch.version.cuda,
  "xformers": bool(importlib.util.find_spec("xformers")),
  "numpy": numpy.__version__,
  "opencv": cv2.__version__,
  "platform": platform.platform()
}, indent=2))
PY

echo "=== Setup Complete ==="
