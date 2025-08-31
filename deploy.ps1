# 🚀 Activate virtual environment
$venvPath = "F:\SoftwareDevelopment\AI Models Image\AIGenerator\.venv\Scripts\Activate.ps1"
Invoke-Expression "& '$venvPath'"

# 🔁 Reinstall PyTorch stack with CUDA 12.4
pip uninstall torch torchvision torchaudio -y
pip install torch==2.8.0+cu124 torchvision==0.23.0+cu124 torchaudio==2.8.0+cu124 `
    --extra-index-url https://download.pytorch.org/whl/cu124

# 📦 Install all dependencies from manifest
pip install -r requirements.txt

# 🧊 Freeze environment for reproducibility
pip freeze > frozen_requirements.txt

# 🧪 Generate and run sanity check script
$sanityCheck = @"
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from realesrgan import RealESRGANer

print("🔧 GPU Available:", torch.cuda.is_available())
print("🧠 CUDA Version:", torch.version.cuda)

try:
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.to("cuda")
    print("✅ Diffusers pipeline loaded")
except Exception as e:
    print("❌ Diffusers error:", e)

try:
    nlp = pipeline("sentiment-analysis")
    print("✅ Transformers pipeline loaded")
except Exception as e:
    print("❌ Transformers error:", e)

try:
    upsampler = RealESRGANer(scale=2, model_path=None, model=None)
    print("✅ Real-ESRGAN initialized")
except Exception as e:
    print("❌ Real-ESRGAN error:", e)
"@
Set-Content -Path sanity_check.py -Value $sanityCheck
python sanity_check.py
