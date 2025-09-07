import os
import torch
import numpy as np
from pathlib import Path
import importlib.util
from datetime import datetime
from PIL import Image

# Telemetry import (fix path if needed)
try:
    from src.modules.utils.telemetry import confirm_launch, confirm_close, log_event
except ImportError:
    confirm_launch = confirm_close = log_event = lambda *a, **k: None

# --- Model weights directory ---
# Use absolute path for your setup (matches your tree on F:)
UPSCALE_WEIGHTS_DIR = Path(r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\upscaler\Real-ESRGAN\weights")

# Supported models (add more as needed)
UPSCALE_MODEL_PATHS = {
    "UltraSharp": UPSCALE_WEIGHTS_DIR / "4x-UltraSharp.pth",
    "Remacri": UPSCALE_WEIGHTS_DIR / "4x_foolhardy_Remacri.pth",
    "Anime6B": UPSCALE_WEIGHTS_DIR / "RealESRGAN_x4plus_anime_6B.pth"
}

MODEL_INTERNAL = {
    "RealESRGAN_x4plus": "RealESRGAN_x4plus",
    "RealESRGAN_x4plus_anime_6B": "RealESRGAN_x4plus_anime_6B",
    "RealESRGAN_x2plus": "RealESRGAN_x2plus",
    "Remacri": "RealESRGAN_x4plus"
}

# --- RRDBNet import ---
# Repo root is the parent of the weights directory:
# ...\models\upscaler\Real-ESRGAN\weights -> repo root = ...\models\upscaler\Real-ESRGAN
repo_root = UPSCALE_WEIGHTS_DIR.parent
rrdb_path = repo_root / "realesrgan" / "archs" / "rrdbnet_arch.py"
if not rrdb_path.exists():
    raise FileNotFoundError(f"RRDBNet arch not found at: {rrdb_path}")

spec = importlib.util.spec_from_file_location("rrdbnet", str(rrdb_path))
rrdbnet = importlib.util.module_from_spec(spec)
assert spec and spec.loader, "Failed to create import spec for rrdbnet"
spec.loader.exec_module(rrdbnet)

RRDBNet = rrdbnet.RRDBNet
remap_checkpoint_keys = getattr(rrdbnet, "remap_checkpoint_keys", lambda x, y: x)

class ESRGANUpscaler:
    def __init__(self, model_path, scale, device, model_name="Unknown"):
        self.model_path = model_path
        self.device = device
        self.scale = scale
        self.model_name = model_name

        print(f"[Init] 🔍 Preparing model '{self.model_name}' from {self.model_path}")
        print(f"[Init] 🎯 Target device: {self.device}")

        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            scale=self.scale
        )

        loadnet = torch.load(self.model_path, map_location=self.device)

        import hashlib
        model_hash = hashlib.md5(open(self.model_path, 'rb').read()).hexdigest()
        print(f"[Audit] 🔐 Model hash: {model_hash}")

        loadnet = remap_checkpoint_keys(loadnet, self.model_path)

        print("[Debug] Sample checkpoint keys:")
        for i, k in enumerate(loadnet.keys()):
            if i < 10:
                print(f"  • {k}")

        try:
            self.model.load_state_dict(loadnet, strict=True)
            print("[Loader] ✅ State dict loaded successfully")
        except RuntimeError as e:
            print(f"[Loader ⚠️] Strict load failed: {e}")
            print("[Telemetry ⚠️] Fallback load may cause visual distortion")
            self.model.load_state_dict(loadnet, strict=False)

        self.model.eval()
        self.model.to(self.device)
        print(f"[Inject ✅] RRDBNet '{self.model_name}' ready on {self.device}")

    def enhance(self, img_np, outscale=4):
        def balance_channels(img_np, strength=0.95):
            mean = img_np.mean(axis=(0, 1))
            scale = mean.max() / (mean + 1e-5)
            return np.clip(img_np * scale * strength, 0, 255).astype(np.uint8)

        with torch.no_grad():
            img_tensor = torch.from_numpy(img_np).float().div(255).permute(2, 0, 1).unsqueeze(0).to(self.device)
            output = self.model(img_tensor).clamp(0, 1)
            output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
            output_np = np.clip(output_np, 0, 1) * 255
            output_np = balance_channels(output_np)
            return output_np.astype(np.uint8), None

def upscale_image_pass(image, model_name="RealESRGAN_x4plus", scale=4, device="cuda", save=True):
    if model_name not in UPSCALE_MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = UPSCALE_MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"[Audit] Upscale model missing → {model_path}")
    model_path = str(model_path)
    output_dir = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\output\upscaled"
    os.makedirs(output_dir, exist_ok=True)

    img_np = np.array(image.convert("RGB"))
    print(f"[Upscaler] 🚀 Enhancing with {model_name} | Scale: {scale}x | Device: {device}")
    print(f"[Validator] 🧠 Model selected: {model_name}")
    print(f"[Validator] 📦 Path resolved: {model_path}")

    upscaler = ESRGANUpscaler(
        model_path=model_path,
        scale=scale,
        device=torch.device(device),
        model_name=model_name
    )

    upscaled, _ = upscaler.enhance(img_np, outscale=scale)
    if upscaled is None:
        raise RuntimeError(f"[Upscale ❌] Model '{model_name}' returned None")

    if upscaled.ndim == 2:
        print("[Normalize] 🖼️ Converting grayscale to RGB")
        upscaled = np.stack([upscaled]*3, axis=-1)

    try:
        upscaled_pil = Image.fromarray(upscaled)
    except Exception as e:
        raise RuntimeError(f"[Upscale ❌] Failed to convert output to PIL: {e}")

    filename = f"upscaled_{model_name}_{scale}x_{int(torch.randint(1000, (1,)).item())}.png"
    save_path = os.path.join(output_dir, filename)

    if save:
        upscaled_pil.save(save_path)
        log_event({
            "event": "UpscalePass",
            "model": model_name,
            "path": str(save_path),
            "timestamp": datetime.now().isoformat()
        })

    return {
        "image": upscaled_pil,
        "filename": filename,
        "path": save_path
    }

def upscale_image_by_path(image, model_path: str, scale=4, device="cuda", save=True):
    """
    Upscale using a direct checkpoint path (.pth). Returns PIL.Image.
    Saves to output/upscaled by default.
    """
    output_dir = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\output\upscaled"
    os.makedirs(output_dir, exist_ok=True)

    img_np = np.array(image.convert("RGB"))
    upscaler = ESRGANUpscaler(
        model_path=model_path,
        scale=scale,
        device=torch.device(device),
        model_name=Path(model_path).stem
    )
    upscaled, _ = upscaler.enhance(img_np, outscale=scale)
    if upscaled is None:
        raise RuntimeError(f"[Upscale ❌] Failed on '{model_path}'")

    if upscaled.ndim == 2:
        upscaled = np.stack([upscaled]*3, axis=-1)

    upscaled_pil = Image.fromarray(upscaled)
    if save:
        from datetime import datetime
        filename = f"upscaled_{Path(model_path).stem}_{scale}x_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(output_dir, filename)
        upscaled_pil.save(save_path)
        log_event({
            "event": "UpscalePass",
            "model": Path(model_path).name,
            "path": str(save_path),
            "timestamp": datetime.utcnow().isoformat()
        })
    return upscaled_pil

if __name__ == "__main__":
    print("[Main] Starting test upscale...")

    try:
        test_image_path = str(repo_root / "inputs" / "0014.jpg")
        image = Image.open(test_image_path)

        result = upscale_image_pass(
            image,
            model_name="RealESRGAN_x4plus",
            scale=4,
            device="cuda",
            save=True
        )

        print(f"[Test ✅] Output saved → {result['path']}")
    except Exception as e:
        print(f"[Main ❌] Upscale test failed: {e}")
