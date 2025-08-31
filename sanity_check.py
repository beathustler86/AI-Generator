import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import subprocess
import sys

# Optional: log output to file
# sys.stdout = open("telemetry/sanity_log.txt", "w")

print("🔧 GPU Available:", torch.cuda.is_available())
print("🧠 CUDA Version:", torch.version.cuda)
print("🧮 Torch Version:", torch.__version__)

# 🧼 Dependency health check
try:
    subprocess.run(["pip", "check"], check=True)
    print("✅ Dependencies are healthy")
except subprocess.CalledProcessError:
    print("❌ Dependency issues detected")

# 🔬 Test tensor creation on GPU
try:
    dummy = torch.randn(1, 3, 512, 512).to("cuda")
    print("✅ Torch tensor created on GPU")
except Exception as e:
    print("❌ Torch tensor error:", e)

# 🎨 Test Stable Diffusion pipeline
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        cache_dir="models/"
    )
    pipe.to("cuda")
    print("✅ Diffusers pipeline loaded")
    # Validate model path
    if not Path(pipe.config._name_or_path).exists():
        print("⚠️ Model path not found:", pipe.config._name_or_path)
except Exception as e:
    print("❌ Diffusers error:", e)

# 🧠 Test Transformers pipeline
try:
    nlp = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    result = nlp("This cockpit is legendary.")
    print("✅ Transformers pipeline loaded:", result)
except Exception as e:
    print("❌ Transformers error:", e)

# 🔼 Real-ESRGAN skipped
print("⏭️ Real-ESRGAN test skipped — module not initialized")

# 🖼️ Test image I/O
try:
    img = Image.fromarray(np.uint8(np.random.rand(512, 512, 3) * 255))
    img.save("test_image.png")
    print("✅ PIL image saved")
except Exception as e:
    print("❌ PIL error:", e)
