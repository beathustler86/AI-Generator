import torch

def validate_rrdb_checkpoint(path, required_keys=None, preview=5, silent=False):
	if required_keys is None:
		required_keys = [
			"conv_first.weight",
			"rrdb_blocks.0.rdb1.layers.0.weight",
			"rrdb_blocks.0.rdb2.layers.2.bias",
			"conv_last.weight"
		]

	try:
		ckpt = torch.load(path, map_location="cpu", weights_only=True)
		if isinstance(ckpt, dict):
			if "params_ema" in ckpt:
				weights = ckpt["params_ema"]
			elif "params" in ckpt:
				weights = ckpt["params"]
			else:
				weights = ckpt
		else:
			if not silent:
				print(f"[Validator ❌] Unexpected format: {type(ckpt)}")
			return False

		keys = list(weights.keys())
		missing = [k for k in required_keys if k not in keys]

		if missing:
			if not silent:
				print(f"[Validator ⚠️] Missing keys: {missing}")
			return False

		if not silent:
			print(f"[Validator ✅] Checkpoint valid → {path}")
			print(f"[Validator] 🔍 Key preview: {keys[:preview]}")
		return True

	except Exception as e:
		if not silent:
			print(f"[Validator ❌] Error loading checkpoint: {e}")
		return False


