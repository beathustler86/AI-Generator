import onnxruntime as ort
from datetime import datetime
import json
import numpy as np
from pathlib import Path

# === Dynamic Path Resolution ===
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "sd3_5_tensorrt" / "ONNX"
LOG_PATH = MODEL_DIR / "load_log.txt"

# === Logging ===
def log_event(message):
	LOG_PATH.parent.mkdir(parents=True, exist_ok=True)  # ✅ Ensure ONNX folder exists
	with LOG_PATH.open("a", encoding="utf-8") as log:
		log.write(f"[{datetime.now().isoformat()}] {message}\n")

# === Load Main SD3.5 Model ===
def load_sd35_main():
	model_path = MODEL_DIR / "model.onnx"
	if not model_path.exists():
		log_event("❌ SD3.5 main model.onnx not found")
		return None
	try:
		session = ort.InferenceSession(str(model_path))
		log_event("✅ SD3.5 main model.onnx loaded")
		return session
	except Exception as e:
		log_event(f"⚠️ SD3.5 main model load failed — {str(e)}")
		return None

# === Load Individual ONNX Modules ===
def load_onnx_model(filename):
	model_path = MODEL_DIR / filename
	if not model_path.exists():
		log_event(f"❌ Missing: {filename}")
		return None
	try:
		session = ort.InferenceSession(str(model_path))
		log_event(f"✅ Loaded: {filename}")
		return session
	except Exception as e:
		log_event(f"⚠️ Failed: {filename} — {str(e)}")
		return None

# === Initialize All SD3.5 Modules ===
def initialize_sd35_modules():
	log_event(f"🧠 Available providers: {ort.get_available_providers()}")

	modules = {
		"clip_g": "clip_g/model_optimized.onnx",
		"clip_l": "clip_l/model_optimized.onnx",
		"t5": "t5/model_optimized.onnx",
		"transformer_fp8": "transformer/fp8/model_optimized.onnx",
		"vae": "vae/model_optimized.onnx",
		"vae_encoder": "vae_encoder/model_optimized.onnx"
	}

	sessions = {}
	for key, file in modules.items():
		if key == "transformer_fp8":
			log_event("🔍 Attempting to load transformer_fp8 — may fail due to bfloat16")
		sessions[key] = load_onnx_model(file)

	# === Telemetry Summary ===
	summary = {
		"loaded": [k for k, v in sessions.items() if v],
		"missing": [k for k, v in sessions.items() if v is None],
		"timestamp": datetime.now().isoformat()
	}
	log_event(f"[Summary] {json.dumps(summary)}")

	return sessions

# === Run Inference ===
def run_inference(session, inputs):
	if session is None:
		raise ValueError("Inference session is None — model failed to load.")
	return session.run(None, inputs)

# === Tokenizer Preprocessing ===
def preprocess(prompt, tokenizer):
	try:
		tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
		return {
			"input_ids": tokens["input_ids"].numpy(),
			"attention_mask": tokens["attention_mask"].numpy()
		}
	except Exception as e:
		log_event(f"⚠️ Tokenizer error: {str(e)}")
		return None