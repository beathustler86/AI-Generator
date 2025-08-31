import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog
import random
import time

import torch
from PIL import Image, ImageTk

try:
    from diffusers import (
        StableDiffusionXLPipeline,
        EulerDiscreteScheduler,
        DPMSolverMultistepScheduler,
        PNDMScheduler,
        DDIMScheduler,
        LCMScheduler,
    )
except Exception as e:
    print(f"[Diffusers] Import warning: {e}")
    StableDiffusionXLPipeline = None
    EulerDiscreteScheduler = DPMSolverMultistepScheduler = PNDMScheduler = DDIMScheduler = LCMScheduler = object

def _lazy_load_cosmos():
    """
    Deferred import; returns None if any cosmos dependency fails.
    """
    try:
        from src.nodes.cosmos_text_to_video import CosmosTextToVideo
        return CosmosTextToVideo()
    except Exception as e:
        print(f"[Cosmos] Unavailable: {e}")
        return None

from src.modules import refiner_module
from src.modules.refiner_module import refine_image  # noqa
from src.modules.utils.telemetry import log_event

BASE_MODELS_ROOT = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_image"
VIDEO_ROOT = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_video"
MODEL_PRESET_PATHS = {
    "SDXL 1.0": os.path.join(BASE_MODELS_ROOT, "sdxl-base-1.0"),
    "DreamShaperV2": os.path.join(BASE_MODELS_ROOT, "dreamshaper-xl-v2-turbo"),
    "Cosmos": os.path.join(VIDEO_ROOT, "ComfyUI"),
    "SD3.5 TensorRT": os.path.join(BASE_MODELS_ROOT, "sd3_5_tensorrt"),
}
OUTPUT_DIR_DEFAULT = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\outputs"


def get_cuda_vram():
    try:
        props = torch.cuda.get_device_properties(0)
        return torch.cuda.memory_allocated() // 1024**2, props.total_memory // 1024**2
    except Exception:
        return 0, 0


class MainWindow(tk.Frame):
    ASPECT_W = 16
    ASPECT_H = 9

    def __init__(self, root, sd35_sessions=None):
        super().__init__(root)
        self.root = root
        self.sd35_sessions = sd35_sessions or {}
        self.status_text = tk.StringVar(value="Idle")
        self.pipe = None
        self.pipe_loaded_model = None
        self.unet_compiled = False
        self.last_image = None
        self.save_dir = os.path.join(OUTPUT_DIR_DEFAULT, "images")
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_paths = self._curate_models()

        self._configure_styles()
        self._build_layout()

        # Performance toggle state
        self.perf_attention_slicing = tk.BooleanVar(value=True)
        self.perf_vae_slicing = tk.BooleanVar(value=True)
        self.perf_xformers = tk.BooleanVar(value=True)
        self.perf_cpu_offload = tk.BooleanVar(value=False)
        self.perf_compile = tk.BooleanVar(value=False)

        # Memory policy / low VRAM mode
        self.low_vram_mode = tk.BooleanVar(value=True)   # default ON since you are VRAM constrained
        self.aggressive_idle = tk.BooleanVar(value=True) # move UNet off GPU when idle
        self.idle_shrink_delay_ms = 8000                 # shrink 8s after last generation
        self._generating = False
        self._idle_shrink_job = None

        # ############ NEW: Embedding / prompt optimization state ############
        self.cache_prompt_embeddings = tk.BooleanVar(value=True)
        self.extend_prompt = tk.BooleanVar(value=True)
        self.half_embed = tk.BooleanVar(value=True)  # NEW: store embeddings in fp16 when possible
        self._emb_cache = {
            "key": None,
            "prompt_embeds": None,
            "negative_embeds": None,
            "pooled_prompt_embeds": None,
            "pooled_negative_embeds": None,
            "chunks": 0,
            "token_count": 0,
            "truncated": False,
            "build_ms": 0.0,
        }
        self.dark_mode = tk.BooleanVar(value=False)
        self.scaling_factor = tk.DoubleVar(value=1.0)

        # Batch / seed state (missing variables in original snippet)
        self.seed_var = tk.IntVar(value=random.randint(0, 2**31 - 1))
        self.batch_size_var = tk.IntVar(value=1)
        self.randomize_batch_seeds = tk.BooleanVar(value=False)
        self.lock_cfg = tk.BooleanVar(value=False)

        self._build_performance_panel()
        self._build_batch_panel()

        self._schedule_telemetry_loop()
        self.cosmos_node = None  # lazy

        threading.Thread(target=self._background_preflight_and_apply, daemon=True).start()
        threading.Thread(target=self._background_preload_refiner, daemon=True).start()
        # Window close hook
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- Styles / Layout ----------
    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#f3f4f6")
        style.configure("TLabelframe", background="#f3f4f6")
        style.configure("TLabelframe.Label", background="#f3f4f6", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#f3f4f6")
        style.configure("TButton", padding=3)

    def _build_layout(self):
        self.root.title("SDXL Cockpit")
        self.root.minsize(1200, 680)
        self.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.left_panel = tk.Frame(self, bg="#f3f4f6")
        self.center_panel = tk.Frame(self, bg="#e9eaed")
        self.left_panel.grid(row=0, column=0, sticky="ns")
        self.center_panel.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.center_panel.columnconfigure(0, weight=1)
        self.center_panel.rowconfigure(0, weight=1)

        self._build_controls(self.left_panel)
        self._build_preview(self.center_panel)
        self._build_bottom_bar()

    def _build_controls(self, parent):
        pad = {'padx': 8, 'pady': 4}
        tk.Label(parent, text="Prompt", anchor="w", font=("Segoe UI", 9, "bold"), bg="#f3f4f6").pack(fill="x", **pad)
        self.prompt_entry = tk.Text(parent, height=5, wrap="word")
        self.prompt_entry.pack(fill="x", padx=8)

        tk.Label(parent, text="Negative Prompt", anchor="w", font=("Segoe UI", 9, "bold"), bg="#f3f4f6").pack(fill="x", **pad)
        self.negative_prompt_entry = tk.Text(parent, height=3, wrap="word")
        self.negative_prompt_entry.pack(fill="x", padx=8)

        ttk.Separator(parent).pack(fill="x", padx=8, pady=6)

        # Model frame
        mf = tk.Frame(parent, bg="#f3f4f6")
        mf.pack(fill="x", **pad)
        tk.Label(mf, text="Model:", bg="#f3f4f6").grid(row=0, column=0, sticky="w")
        self.selected_model = tk.StringVar(value=(next(iter(self.model_paths)) if self.model_paths else ""))
        self.model_menu = ttk.Combobox(mf, textvariable=self.selected_model,
                                       values=list(self.model_paths.keys()), state="readonly", width=20)
        self.model_menu.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        tk.Button(mf, text="Load", command=self.load_selected_model, width=6).grid(row=0, column=2, padx=6)
        mf.columnconfigure(1, weight=1)

        # Resolution
        res = tk.LabelFrame(parent, text="Resolution / Aspect", padx=6, pady=4)
        res.pack(fill="x", padx=8, pady=(2, 4))
        self.width_var = tk.IntVar(value=1280)
        self.height_var = tk.IntVar(value=720)
        tk.Label(res, text="W").grid(row=0, column=0, sticky="e")
        tk.Entry(res, textvariable=self.width_var, width=6).grid(row=0, column=1, padx=(2, 6))
        tk.Label(res, text="H").grid(row=0, column=2, sticky="e")
        tk.Entry(res, textvariable=self.height_var, width=6).grid(row=0, column=3, padx=(2, 6))
        tk.Button(res, text="16:9", command=lambda: self._apply_aspect(16, 9), width=5).grid(row=1, column=0)
        tk.Button(res, text="1:1", command=lambda: self._apply_aspect(1, 1), width=5).grid(row=1, column=1)
        tk.Button(res, text="4:3", command=lambda: self._apply_aspect(4, 3), width=5).grid(row=1, column=2)
        tk.Button(res, text="Apply", command=self._apply_custom_resolution, width=7).grid(row=1, column=3)

        # Steps / Guidance
        sg = tk.LabelFrame(parent, text="Steps / Guidance", padx=6, pady=4)
        sg.pack(fill="x", padx=8, pady=(2, 4))
        tk.Label(sg, text="Steps").grid(row=0, column=0, sticky="w")
        self.steps_var = tk.IntVar(value=30)
        tk.Scale(sg, from_=5, to=150, orient="horizontal", variable=self.steps_var,
                 showvalue=True, length=180).grid(row=0, column=1, columnspan=3, sticky="ew", padx=(4, 0))
        tk.Label(sg, text="Guidance").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.guidance_var = tk.DoubleVar(value=7.5)
        tk.Scale(sg, from_=1.0, to=20.0, resolution=0.1, orient="horizontal",
                 variable=self.guidance_var, showvalue=True, length=180)\
            .grid(row=1, column=1, columnspan=3, sticky="ew", padx=(4, 0), pady=(6, 0))

        # Sampler
        sampler_f = tk.Frame(parent, bg="#f3f4f6")
        sampler_f.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(sampler_f, text="Sampler:", bg="#f3f4f6").grid(row=0, column=0, sticky="w")
        self.sampler_var = tk.StringVar(value="Euler")
        self.sampler_menu = ttk.Combobox(
            sampler_f, textvariable=self.sampler_var,
            values=["Euler", "DPM++", "PNDM", "DDIM", "LCM"],
            state="readonly", width=10
        )
        self.sampler_menu.grid(row=0, column=1, padx=4)

        # Action buttons
        act = tk.Frame(parent, bg="#f3f4f6")
        act.pack(fill="x", padx=8, pady=(8, 4))
        tk.Button(act, text="Generate", command=self.threaded_generate, width=10).grid(row=0, column=0, padx=2)
        self.refine_button = tk.Button(act, text="Refine", command=self.refine_current, width=8, state="disabled")
        self.refine_button.grid(row=0, column=1, padx=2)
        tk.Button(act, text="Video", command=self.generate_video, width=8).grid(row=0, column=2, padx=2)
        tk.Button(act, text="Upload", command=self.upload_image, width=8).grid(row=0, column=3, padx=2)
        tk.Button(act, text="Choose Dir", command=self.choose_save_dir, width=10).grid(row=1, column=0, padx=2, pady=(6, 0))
        tk.Button(act, text="Save Image", command=self.save_current_image, width=10).grid(row=1, column=1, padx=2, pady=(6, 0))
        act.columnconfigure(4, weight=1)

    def _build_performance_panel(self):
        pf = tk.LabelFrame(self.left_panel, text="Performance", padx=6, pady=4)
        pf.pack(fill="x", padx=8, pady=(2, 4))
        tk.Checkbutton(pf, text="Attention slicing", variable=self.perf_attention_slicing, bg="#f3f4f6").grid(row=0, column=0, sticky="w")
        tk.Checkbutton(pf, text="VAE slicing", variable=self.perf_vae_slicing, bg="#f3f4f6").grid(row=1, column=0, sticky="w")
        tk.Checkbutton(pf, text="xFormers (if avail)", variable=self.perf_xformers, bg="#f3f4f6").grid(row=2, column=0, sticky="w")
        tk.Checkbutton(pf, text="CPU offload", variable=self.perf_cpu_offload, bg="#f3f4f6").grid(row=3, column=0, sticky="w")
        tk.Checkbutton(pf, text="Compile UNet (PyTorch 2)", variable=self.perf_compile, bg="#f3f4f6").grid(row=4, column=0, sticky="w")
        tk.Button(pf, text="Apply Perf", command=self.apply_performance_toggles, width=12).grid(row=5, column=0, pady=(6, 0))
        # Low VRAM box
        lv = tk.LabelFrame(self.left_panel, text="Memory", padx=6, pady=4)
        lv.pack(fill="x", padx=8, pady=(2, 6))
        tk.Checkbutton(lv, text="Low VRAM mode", variable=self.low_vram_mode, bg="#f3f4f6",
                       command=lambda: self.update_telemetry_status("LowVRAM ON" if self.low_vram_mode.get() else "LowVRAM OFF")).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(lv, text="Aggressive idle shrink", variable=self.aggressive_idle, bg="#f3f4f6").grid(row=1, column=0, sticky="w")
        tk.Button(lv, text="Shrink Now", command=self.force_idle_shrink, width=10).grid(row=2, column=0, pady=(4, 0))
        tk.Button(lv, text="Deep Reset", command=self.deep_vram_reset, width=10).grid(row=3, column=0, pady=(4, 0))

        # ############ NEW: Prompt / UI box ############
        embf = tk.LabelFrame(self.left_panel, text="Prompt / UI", padx=6, pady=4)
        embf.pack(fill="x", padx=8, pady=(2, 8))
        tk.Checkbutton(embf, text="Cache embeddings", variable=self.cache_prompt_embeddings, bg="#f3f4f6").grid(row=0, column=0, sticky="w")
        tk.Checkbutton(embf, text="Extended prompt (>77)", variable=self.extend_prompt, bg="#f3f4f6").grid(row=1, column=0, sticky="w")
        tk.Checkbutton(embf, text="Half embeds", variable=self.half_embed, bg="#f3f4f6").grid(row=2, column=0, sticky="w")
        tk.Checkbutton(embf, text="Dark mode", variable=self.dark_mode, bg="#f3f4f6",
                       command=self._toggle_dark_mode).grid(row=3, column=0, sticky="w")
        tk.Label(embf, text="Scaling", bg="#f3f4f6").grid(row=4, column=0, sticky="w")
        tk.Scale(embf, from_=0.75, to=1.75, resolution=0.05, orient="horizontal",
                 variable=self.scaling_factor, length=140,
                 command=lambda _v: self._apply_scaling()).grid(row=5, column=0, sticky="ew")
        self.prompt_diag_var = tk.StringVar(value="Tokens: -")
        tk.Label(embf, textvariable=self.prompt_diag_var, bg="#f3f4f6", fg="#555",
                 font=("Segoe UI", 8)).grid(row=6, column=0, sticky="w", pady=(4,0))

    def _build_batch_panel(self):
        bf = tk.LabelFrame(self.left_panel, text="Batch / Seed", padx=6, pady=4)
        bf.pack(fill="x", padx=8, pady=(2, 8))
        tk.Label(bf, text="Seed").grid(row=0, column=0, sticky="w")
        tk.Entry(bf, textvariable=self.seed_var, width=14).grid(row=0, column=1, padx=4)
        tk.Button(bf, text="Random", command=self._randomize_seed, width=7).grid(row=0, column=2, padx=2)
        tk.Label(bf, text="Batch").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.Entry(bf, textvariable=self.batch_size_var, width=6).grid(row=1, column=1, padx=4, pady=(6, 0))
        tk.Checkbutton(bf, text="Randomize per image", variable=self.randomize_batch_seeds, bg="#f3f4f6").grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        tk.Checkbutton(bf, text="Lock CFG", variable=self.lock_cfg, bg="#f3f4f6").grid(row=3, column=0, columnspan=3, sticky="w")

    def _build_preview(self, parent):
        self.preview_frame = tk.Frame(parent, bg="#e9eaed")
        self.preview_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.preview_frame, bg="#111", highlightthickness=1, highlightbackground="#444")
        self.canvas.place(relx=0.5, rely=0.5, anchor="center", width=960, height=540)
        parent.bind("<Configure>", self._on_center_resize)

    def _build_bottom_bar(self):
        bar = tk.Frame(self, bd=1, relief="sunken", bg="#f3f4f6")
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.progress_bar = ttk.Progressbar(bar, orient="horizontal", mode="determinate", length=240)
        self.progress_bar.pack(side="right", padx=10, pady=4)
        self.telemetry_label = tk.Label(bar, textvariable=self.status_text, anchor="e", width=28, bg="#f3f4f6")
        self.telemetry_label.pack(side="right", padx=10)

    # ---------- Telemetry loop ----------
    def _schedule_telemetry_loop(self):
        self.update_telemetry_status()
        self.root.after(2500, self._schedule_telemetry_loop)

    # ---------- Resize & display ----------
    def _on_center_resize(self, event):
        w, h = event.width, event.height
        if w < 20 or h < 20:
            return
        max_w, max_h = int(w * 0.95), int(h * 0.95)
        tw = max_w
        th = int(tw * self.ASPECT_H / self.APECT_W)
        if th > max_h:
            th = max_h
            tw = int(th * self.ASPECT_W / self.ASPECT_H)
        self.canvas.place_configure(width=tw, height=th)
        if self.last_image is not None:
            self._display_image(self.last_image)

    def _display_image(self, pil_img):
        cw = int(self.canvas.winfo_width())
        ch = int(self.canvas.winfo_height())
        if cw < 10 or ch < 10:
            return
        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih)
        disp = pil_img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

    # ---------- Models / Preflight ----------
    def _curate_models(self):
        return {n: p for n, p in MODEL_PRESET_PATHS.items() if os.path.exists(p)}

    def _background_preflight_and_apply(self):
        try:
            from src.modules.preflight_check import run_preflight
            result = run_preflight()
            missing = result.get("missing_files") if isinstance(result, dict) else []
            self.update_telemetry_status("Preflight OK" if not missing else f"Preflight: {len(missing)} missing")
        except Exception as e:
            self.update_telemetry_status(f"Preflight error: {e}")

    def _background_preload_refiner(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            refiner_module.load_refiner(device=device, force_reload=False)
            log_event({"event": "RefinerWarmup", "device": device, "timestamp": datetime.now().isoformat()})
        except Exception as e:
            log_event({"event": "RefinerWarmupFailed", "error": str(e), "timestamp": datetime.now().isoformat()})

    # ---------- Aspect / Resolution ----------
    def _apply_aspect(self, aw, ah):
        h = self.height_var.get()
        w = int(round(h * aw / ah))
        self.width_var.set(w)
        self.ASPECT_W, self.ASPECT_H = aw, ah
        self.update_telemetry_status(f"Aspect {aw}:{ah}")

    def _apply_custom_resolution(self):
        w, h = self.width_var.get(), self.height_var.get()
        if h > 0:
            from math import gcd
            g = gcd(w, h)
            self.ASPECT_W, self.ASPECT_H = w // g, h // g
        self.update_telemetry_status(f"{w}x{h}")

    # ---------- Load Pipeline ----------
    def load_selected_model(self):
        name = self.selected_model.get()
        path = self.model_paths.get(name)
        if not path or StableDiffusionXLPipeline is None:
            self.update_telemetry_status("Model path / diffusers missing.")
            return
        if self.pipe_loaded_model == name:
            self.update_telemetry_status("Already loaded.")
            return
        self.update_telemetry_status(f"Loading {name}...")
        threading.Thread(target=self._load_model_thread, args=(name, path), daemon=True).start()

    def _load_model_thread(self, name, path):
        try:
            t0 = time.perf_counter()
            torch.set_float32_matmul_precision("high")
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            kwargs = dict(torch_dtype=dtype, local_files_only=True)
            if os.path.isdir(path):
                pipe = StableDiffusionXLPipeline.from_pretrained(path, **kwargs)
            else:
                pipe = StableDiffusionXLPipeline.from_single_file(path, **kwargs)

            if self.perf_attention_slicing.get() and hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if self.perf_vae_slicing.get() and hasattr(pipe, "enable_vae_slicing"):
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass
            if self.perf_xformers.get():
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            if hasattr(pipe, "safety_checker"):
                try:
                    pipe.safety_checker = None
                except Exception:
                    pass
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe.to(device)

            if self.perf_cpu_offload.get() and hasattr(pipe, "enable_model_cpu_offload") and device == "cuda":
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass

            if self.perf_compile.get() and not self.unet_compiled and hasattr(torch, "compile") and hasattr(pipe, "unet"):
                try:
                    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
                    self.unet_compiled = True
                except Exception:
                    pass

            self.pipe = pipe
            self.pipe_loaded_model = name
            load_ms = int((time.perf_counter() - t0) * 1000)
            self.update_telemetry_status(f"Loaded {name} ({load_ms}ms)")

            def _warm():
                try:
                    if self.low_vram_mode.get():
                        return
                    if torch.cuda.is_available():
                        with torch.autocast("cuda"):
                            pipe(prompt="warmup", num_inference_steps=4, width=512, height=512)
                    else:
                        pipe(prompt="warmup", num_inference_steps=2, width=256, height=256)
                except Exception:
                    pass
            threading.Thread(target=_warm, daemon=True).start()
            if self.low_vram_mode.get():
                self._shrink_pipeline_to_cpu(initial=True)
        except Exception as e:
            self.update_telemetry_status(f"Load failed: {e}")

    def apply_performance_toggles(self):
        if not self.pipe:
            self.update_telemetry_status("Load model first.")
            return
        try:
            if self.perf_attention_slicing.get() and hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            if not self.perf_attention_slicing.get() and hasattr(self.pipe, "disable_attention_slicing"):
                try:
                    self.pipe.disable_attention_slicing()
                except Exception:
                    pass

            if self.perf_vae_slicing.get() and hasattr(self.pipe, "enable_vae_slicing"):
                try:
                    self.pipe.enable_vae_slicing()
                except Exception:
                    pass

            if self.perf_xformers.get():
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

            if self.perf_cpu_offload.get() and hasattr(self.pipe, "enable_model_cpu_offload"):
                try:
                    self.pipe.enable_model_cpu_offload()
                except Exception:
                    pass

            if self.perf_compile.get() and hasattr(torch, "compile") and hasattr(self.pipe, "unet") and not self.unet_compiled:
                try:
                    self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
                    self.unet_compiled = True
                except Exception:
                    pass
            self.update_telemetry_status("Perf applied")
        except Exception as e:
            self.update_telemetry_status(f"Perf error: {e}")

    # ---------- Generation ----------
    def _get_scheduler_class(self, name):
        return {
            "Euler": EulerDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM": DDIMScheduler,
            "LCM": LCMScheduler,
        }.get(name)

    def threaded_generate(self):
        threading.Thread(target=self.generate_image, daemon=True).start()

    def _start_progress(self):
        try:
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start(12)
        except Exception:
            pass

    def _stop_progress(self):
        try:
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", value=0)
        except Exception:
            pass

    def generate_image(self):
        if self.pipe is None:
            self.update_telemetry_status("No pipeline.")
            return

        # Collect current UI values early (fix: previously used before assignment)
        prompt = self.prompt_entry.get("1.0", "end").strip()
        negative = self.negative_prompt_entry.get("1.0", "end").strip()
        width, height = self.width_var.get(), self.height_var.get()
        steps, guidance = self.steps_var.get(), self.guidance_var.get()
        batch_size = max(1, self.batch_size_var.get())
        base_seed = self.seed_var.get()

        # Cancel pending shrink while generating
        if self._idle_shrink_job:
            self.root.after_cancel(self._idle_shrink_job)
            self._idle_shrink_job = None
        self._generating = True
        if self.low_vram_mode.get():
            self._stage_pipeline_on_gpu()

        sched_cls = self._get_scheduler_class(self.sampler_var.get())
        if sched_cls:
            try:
                if not isinstance(self.pipe.scheduler, sched_cls):
                    self.pipe.scheduler = sched_cls.from_config(self.pipe.scheduler.config)
            except Exception:
                pass

        self._start_progress()
        self.update_telemetry_status("Generating...")
        images = []
        try:
            torch.set_grad_enabled(False)
            device_autocast = torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad()

            # Prompt embedding cache
            use_embed = False
            cached_embeds = None
            if self.pipe and hasattr(self.pipe, "tokenizer") and (self.cache_prompt_embeddings.get() or self.extend_prompt.get()):
                cached_embeds = self._get_cached_embeddings(prompt, negative, guidance)
                if cached_embeds and cached_embeds["prompt_embeds"] is not None:
                    use_embed = True
                    self.prompt_diag_var.set(
                        f"Tokens:{cached_embeds['token_count']}  Chunks:{cached_embeds['chunks']}  Build:{cached_embeds['build_ms']}ms" +
                        (" TRUNC" if cached_embeds["truncated"] else "")
                    )
                else:
                    self.prompt_diag_var.set("Tokens: -")

            for i in range(batch_size):
                seed = (base_seed + i) if self.randomize_batch_seeds.get() else base_seed
                g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
                with device_autocast:
                    if use_embed:
                        try:
                            result = self.pipe(
                                prompt=None,
                                prompt_embeds=cached_embeds["prompt_embeds"],
                                negative_prompt=None,
                                negative_prompt_embeds=cached_embeds["negative_embeds"],
                                pooled_prompt_embeds=cached_embeds["pooled_prompt_embeds"],
                                negative_pooled_prompt_embeds=cached_embeds["pooled_negative_embeds"],
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                width=width,
                                height=height,
                                generator=g,
                            )
                        except Exception as embed_err:
                            print(f"[EmbedFallback] {embed_err}")
                            use_embed = False
                            result = self.pipe(
                                prompt=prompt,
                                negative_prompt=negative,
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                width=width,
                                height=height,
                                generator=g,
                            )
                    else:
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=negative,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            width=width,
                            height=height,
                            generator=g,
                        )
                if not result or not getattr(result, "images", None):
                    continue
                img = result.images[0]
                images.append((seed, img))
                if i == 0:
                    self.last_image = img
                    self._display_image(img)
                try:
                    del result
                except Exception:
                    pass

            if not images:
                self.update_telemetry_status("No output.")
                return
            os.makedirs(self.save_dir, exist_ok=True)
            for seed, img in images:
                img.save(os.path.join(self.save_dir, f"generated_seed{seed}.png"))
            self.refine_button.config(state="normal")
            self.update_telemetry_status(f"Done ({len(images)})")
        except Exception as e:
            self.update_telemetry_status(f"Error: {e}")
        finally:
            self._generating = False
            if self.low_vram_mode.get():
                self._idle_shrink_job = self.root.after(self.idle_shrink_delay_ms, self._shrink_pipeline_to_cpu)
            self._stop_progress()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def refine_current(self):
        if self.last_image is None:
            self.update_telemetry_status("No image to refine.")
            return
        try:
            out = refine_image(self.last_image)
            self.last_image = out
            self._display_image(out)
            out.save(os.path.join(self.save_dir, "refined_latest.png"))
            self.update_telemetry_status("Refined.")
        except Exception as e:
            self.update_telemetry_status(f"Refine failed: {e}")

    def generate_video(self):
        if self.cosmos_node is None:
            self.cosmos_node = _lazy_load_cosmos()
        if not self.cosmos_node:
            self.update_telemetry_status("Cosmos disabled.")
            return
        prompt = self.prompt_entry.get("1.0", "end").strip()
        self.update_telemetry_status("Cosmos...")

        def _run():
            try:
                frame0, _ = self.cosmos_node.generate(prompt, frame_count=8, width=512, height=512)
                img = Image.fromarray(frame0)
                self.last_image = img
                self._display_image(img)
                self.update_telemetry_status("Cosmos OK.")
            except Exception as e:
                self.update_telemetry_status(f"Cosmos error: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # ---------- Saving / Upload ----------
    def choose_save_dir(self):
        d = filedialog.askdirectory(title="Choose Save Directory")
        if d:
            self.save_dir = d
            self.update_telemetry_status("Dir set")

    def save_current_image(self):
        if self.last_image is None:
            self.update_telemetry_status("No image.")
            return
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.save_dir, f"image_{ts}.png")
            self.last_image.save(path)
            self.update_telemetry_status(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.update_telemetry_status(f"Save failed: {e}")

    def upload_image(self):
        fp = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")],
        )
        if not fp:
            return
        try:
            img = Image.open(fp).convert("RGB")
            self.last_image = img
            self._display_image(img)
            self.refine_button.config(state="normal")
            self.update_telemetry_status(f"Loaded {os.path.basename(fp)}")
        except Exception as e:
            self.update_telemetry_status(f"Upload err: {e}")

    # ---------- Utilities ----------
    def _randomize_seed(self):
        self.seed_var.set(random.randint(0, 2**31 - 1))
        self.update_telemetry_status("Seed randomized")

    def update_telemetry_status(self, msg=None):
        if msg:
            self.status_text.set(msg)
        else:
            used, total = get_cuda_vram()
            self.status_text.set(f"VRAM {used}/{total}MB" if total else "CPU Mode")

    # ---------- Memory management helpers ----------
    def _pipeline_parts(self):
        if not self.pipe:
            return []
        parts = []
        for attr in ("text_encoder", "text_encoder_2", "vae", "unet"):
            m = getattr(self.pipe, attr, None)
            if m is not None:
                parts.append((attr, m))
        return parts

    def _stage_pipeline_on_gpu(self):
        if not self.pipe or not torch.cuda.is_available():
            return
        for name, module in self._pipeline_parts():
            if name in ("unet", "vae") or name.startswith("text_encoder"):
                try:
                    module.to("cuda")
                except Exception:
                    pass
        try:
            if self.perf_attention_slicing.get() and hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
        except Exception:
            pass

    def _shrink_pipeline_to_cpu(self, initial=False):
        if not self.pipe or self._generating:
            return
        aggressive = self.aggressive_idle.get()
        moved = []
        for name, module in self._pipeline_parts():
            if not aggressive and name == "unet":
                continue
            try:
                module.to("cpu")
                moved.append(name)
            except Exception:
                pass
        try:
            if self.pipe and hasattr(self.pipe, "scheduler"):
                for n, m in self.pipe.scheduler.__dict__.items():
                    if hasattr(m, "to") and hasattr(m, "parameters"):
                        try:
                            m.to("cpu")
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        if not initial:
            self.update_telemetry_status(f"Shrunk ({','.join(moved)})")

    def force_idle_shrink(self):
        self._shrink_pipeline_to_cpu()

    # ---------- NEW: Prompt embedding cache / extended encoding ----------
    def _get_cached_embeddings(self, prompt: str, negative: str, guidance_scale: float):
        if not self.pipe or not hasattr(self.pipe, "tokenizer"):
            return None
        pipe = self.pipe
        tokenizer = pipe.tokenizer
        try:
            max_len = getattr(tokenizer, "model_max_length", 77)
        except Exception:
            max_len = 77

        key = f"{prompt}|{negative}|ext:{self.extend_prompt.get()}|g:{guidance_scale>1.0}"
        if self.cache_prompt_embeddings.get() and self._emb_cache["key"] == key:
            return self._emb_cache

        t_start = time.perf_counter()
        try:
            tokens_full = tokenizer(prompt, return_tensors="pt", truncation=False, padding=False).input_ids[0]
            token_count = tokens_full.shape[0]
            segments = [prompt]
            truncated_flag = False

            if self.extend_prompt.get() and token_count > max_len:
                parts = [p.strip() for p in prompt.replace("\n", " ").split(",") if p.strip()]
                segments = []
                cur_text = ""
                for p in parts:
                    candidate = (cur_text + ", " + p) if cur_text else p
                    cand_ids = tokenizer(candidate, return_tensors="pt", truncation=False).input_ids[0]
                    if cand_ids.shape[0] <= max_len - 2:
                        cur_text = candidate
                    else:
                        if cur_text:
                            segments.append(cur_text)
                        # Hard chunk p if needed
                        long_ids = tokenizer(p, return_tensors="pt", truncation=False).input_ids[0]
                        while long_ids.shape[0] > max_len - 2:
                            chunk_ids = long_ids[: max_len - 2]
                            segments.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
                            long_ids = long_ids[max_len - 2 :]
                        if long_ids.shape[0] > 0:
                            segments.append(tokenizer.decode(long_ids, skip_special_tokens=True))
                        cur_text = ""
                if cur_text:
                    segments.append(cur_text)
            elif token_count > max_len:
                truncated_flag = True  # will rely on internal truncation

            prompt_embeds_list = []
            pooled_list = []
            negative_embeds = None
            negative_pooled = None
            do_guidance = guidance_scale > 1.0

            for idx, seg in enumerate(segments):
                enc = pipe.encode_prompt(
                    prompt=seg,
                    device=pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_guidance,
                    negative_prompt=negative if idx == 0 else None,
                )
                if len(enc) == 4:
                    pe, ne, pp, npool = enc
                else:
                    # Older variant fallback
                    pe = enc[0]
                    ne = enc[1] if len(enc) > 1 else None
                    pp = npool = None
                prompt_embeds_list.append(pe)
                if pp is not None:
                    pooled_list.append(pp)
                if idx == 0:
                    negative_embeds = ne
                    negative_pooled = npool

            # Merge by mean
            if len(prompt_embeds_list) == 1:
                merged_prompt = prompt_embeds_list[0]
            else:
                merged_prompt = torch.stack(prompt_embeds_list, dim=0).mean(0)

            merged_pooled = None
            if pooled_list:
                if len(pooled_list) == 1:
                    merged_pooled = pooled_list[0]
                else:
                    merged_pooled = torch.stack(pooled_list, dim=0).mean(0)

            if self.half_embed.get() and torch.cuda.is_available():
                try:
                    merged_prompt = merged_prompt.to(dtype=torch.float16)
                    if negative_embeds is not None:
                        negative_embeds = negative_embeds.to(dtype=torch.float16)
                    if merged_pooled is not None:
                        merged_pooled = merged_pooled.to(dtype=torch.float16)
                    if negative_pooled is not None:
                        negative_pooled = negative_pooled.to(dtype=torch.float16)
                except Exception:
                    pass

            build_ms = int((time.perf_counter() - t_start) * 1000)
            self._emb_cache = {
                "key": key,
                "prompt_embeds": merged_prompt,
                "negative_embeds": negative_embeds,
                "pooled_prompt_embeds": merged_pooled,
                "pooled_negative_embeds": negative_pooled,
                "chunks": len(segments),
                "token_count": token_count,
                "truncated": truncated_flag and not self.extend_prompt.get(),
                "build_ms": build_ms,
            }
            # Free lists
            del prompt_embeds_list
            pooled_list.clear()
            return self._emb_cache
        except Exception as e:
            print(f"[EmbCacheError] {e}")
            return None

    # ---------- NEW: Dark / Light toggle & scaling ----------
    def _toggle_dark_mode(self):
        try:
            if self.dark_mode.get():
                self.root.configure(bg="#202224")
            else:
                self.root.configure(bg="#f0f0f0")
        except Exception:
            pass

    def _apply_scaling(self):
        try:
            self.root.tk.call('tk', 'scaling', self.scaling_factor.get())
        except Exception:
            pass

    # ---------- Diagnostics / Deep cleanup ----------
    def _print_vram(self, label="VRAM"):
        if not torch.cuda.is_available():
            return
        try:
            alloc = torch.cuda.memory_allocated()//1024//1024
            reserved = torch.cuda.memory_reserved()//1024//1024
            print(f"[{label}] allocated={alloc}MB reserved={reserved}MB")
        except Exception:
            pass

    def deep_vram_reset(self):
        """
        Force aggressive module removal and cache purge.
        """
        self._print_vram("BeforeDeepReset")
        try:
            if self.pipe:
                for name, module in self._pipeline_parts():
                    try:
                        module.to("cpu")
                    except Exception:
                        pass
                if hasattr(self.pipe, "scheduler"):
                    try:
                        self.pipe.scheduler = None
                    except Exception:
                        pass
                self.pipe = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"[DeepResetError] {e}")
        self._print_vram("AfterDeepReset")
        self.update_telemetry_status("Deep reset done")

    def on_close(self):
        """
        Graceful teardown to ensure CUDA contexts release quickly.
        """
        try:
            self._generating = False
            if self._idle_shrink_job:
                self.root.after_cancel(self._idle_shrink_job)
            self.deep_vram_reset()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass
logger.info("[Launcher] Launch sequence started")
















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































