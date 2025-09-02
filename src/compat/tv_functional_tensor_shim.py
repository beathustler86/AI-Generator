import sys, types, importlib, importlib.abc, importlib.util, importlib.machinery

TARGET = "torchvision.transforms.functional_tensor"


def _export_into(module):
    import torchvision.transforms.functional as F
    for k, v in F.__dict__.items():
        if not k.startswith("_"):
            setattr(module, k, v)
    if hasattr(F, "rgb_to_grayscale"):
        module.rgb_to_grayscale = F.rgb_to_grayscale


class _FunctionalTensorLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None  # default behavior
    def exec_module(self, module):
        _export_into(module)


class _FunctionalTensorFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == TARGET:
            existing = sys.modules.get(fullname)
            if existing and getattr(existing, "__spec__", None):
                return existing.__spec__
            return importlib.machinery.ModuleSpec(
                name=fullname,
                loader=_FunctionalTensorLoader(),
                origin="shim:functional_tensor_removed"
            )
        return None


def install():
    if TARGET in sys.modules:
        return True
    # If an original module exists (older torchvision), leave it alone
    try:
        if importlib.util.find_spec(TARGET) is not None:
            return True
    except Exception:
        pass
    # Trigger loader through normal import (finder will supply spec)
    try:
        importlib.import_module(TARGET)
        return True
    except Exception:
        return False


def ensure_installed(verbose: bool = False):
    if not any(isinstance(f, _FunctionalTensorFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FunctionalTensorFinder())
    ok = install()
    if verbose and ok and TARGET in sys.modules:
        print(f"[Compat] shim active: {TARGET}", flush=True)
    return ok


# Execute on import
ensure_installed()

if __name__ == "__main__":
    ensure_installed()
    import importlib
    ok = bool(importlib.util.find_spec(TARGET))
    print("Shim ready:", ok)
