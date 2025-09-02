Param(
    [string]$PythonExe = "python",
    [switch]$Dev
)
Write-Host "=== AI Generator Environment Setup ==="

if (-not (Test-Path .\.venv)) {
    & $PythonExe -m venv .venv
}
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

# Install numpy first so torch doesn't drag in numpy 2.x then downgrade
pip install numpy==1.26.4 opencv-python==4.10.0.84

pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 `
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124

pip install -r requirements.txt -c constraints-gpu.txt --upgrade --upgrade-strategy only-if-needed

# xformers intentionally skipped; install manually only if requested
# pip install --extra-index-url https://download.pytorch.org/whl/cu124 xformers==<version>

pip install --no-deps "realesrgan @ git+https://github.com/xinntao/Real-ESRGAN.git@a4abfb2979a7bbff3f69f58f58ae324608821e27"

if ($Dev) {
    pip install -r requirements-dev.txt -c constraints-gpu.txt
}

python -c "import torch,importlib,cv2,numpy,json;import platform;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'xformers':bool(importlib.util.find_spec('xformers')),'numpy':numpy.__version__,'opencv':cv2.__version__,'platform':platform.platform()},indent=2))"
Write-Host '=== Setup Complete ==='
