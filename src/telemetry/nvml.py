try:
    import pynvml
    _init = False
except Exception:
    pynvml = None
    _init = False

def _ensure():
    global _init
    if pynvml and not _init:
        try:
            pynvml.nvmlInit(); _init=True
        except Exception:
            _init=False

def gpu_stats():
    if not pynvml:
        return {}
    _ensure()
    if not _init:
        return {}
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "gpu_util_pct": util.gpu,
            "gpu_mem_used_mb": mem.used//1024//1024,
            "gpu_mem_total_mb": mem.total//1024//1024,
            "gpu_temp_c": temp
        }
    except Exception:
        return {}