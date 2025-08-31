import os
import torch
import numpy as np
import sys
from pathlib import Path
import importlib.util
from datetime import datetime
from PIL import Image
from telemetry import confirm_launch, confirm_close


rrdb_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'Upscaler', 'upscaler-ultra', 'realesrgan', 'archs', 'rrdbnet.py')
spec = importlib.util.spec_from_file_location("rrdbnet", rrdb_path)
rrdbnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rrdbnet)

RRDBNet = rrdbnet.RRDBNet
remap_checkpoint_keys = rrdbnet.remap_checkpoint_keys	# ✅ Unified remap dispatcher

from src.config.config_paths import UPSCALE_MODEL_PATHS

print(f"[DEBUG] UPSCALE_MODEL_PATHS loaded with keys: {list(UPSCALE_MODEL_PATHS.keys())}")  # ✅ Registry confirmed

UPSCALE_MODEL_PATHS = {
	"Remacri": Path("F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/Upscaler/upscaler-ultra/4x_foolhardy_Remacri.pth")
}

MODEL_INTERNAL = {
	"Remacri": "RealESRGAN_x4plus"
}

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

def upscale_image_pass(image, model_name="Remacri", scale=4, device="cuda", save=True):
	model_name = "Remacri"	# 🔒 Force override to prevent GUI/config breach

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
		from src.modules.preflight_check import log_event
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

if __name__ == "__main__":
	print("[Main] Starting test upscale...")

	try:
		test_image_path = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\Upscaler\upscaler-ultra\inputs\0014.jpg"
		image = Image.open(test_image_path)

		result = upscale_image_pass(
			image,
			model_name="Remacri",
			scale=4,
			device="cuda",
			save=True
		)

		print(f"[Test ✅] Output saved → {result['path']}")
	except Exception as e:
		print(f"[Main ❌] Upscale test failed: {e}")