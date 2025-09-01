import sys, importlib
MODS = ["PySide6","torch","diffusers","transformers","PIL","onnx","onnxruntime","realesrgan","gfpgan"]
print("[Env] Python exe:", sys.executable)
for m in MODS:
    print(f"[Env] {m:<12}:", "OK" if importlib.util.find_spec(m) else "MISSING")
