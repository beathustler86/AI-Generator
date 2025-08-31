from datetime import datetime
import importlib
from src.modules.utils.telemetry import log_event

PKGS = ["diffusers", "transformers", "huggingface_hub", "torch"]

def gather_versions(pkgs=PKGS):
	versions = {}
	for name in pkgs:
		try:
			mod = importlib.import_module(name)
			ver = getattr(mod, "__version__", None)
			# fallback to importlib.metadata if available and module has no __version__
			if ver is None:
				try:
					from importlib import metadata as importlib_metadata
					ver = importlib_metadata.version(name)
				except Exception:
					ver = "unknown"
			versions[name] = ver
		except Exception as e:
			versions[name] = f"import_failed: {e}"
	return versions

def run_version_check():
	versions = gather_versions()
	payload = {
		"event": "VersionCheck",
		"versions": versions,
		"timestamp": datetime.now().isoformat()
	}
	# use centralized telemetry (safe, non-blocking)
	log_event(payload)
	print("[Telemetry] Version check logged.")
	return versions

if __name__ == "__main__":
	run_version_check()
