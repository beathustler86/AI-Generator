from __future__ import annotations
from typing import Optional, Any, List, Set
import os, sys, threading, contextlib
from pathlib import Path
import time
import datetime

try:
    from PySide6 import QtWidgets as QtW, QtCore, QtGui
except ImportError:  # pragma: no cover
    from PyQt5 import QtWidgets as QtW, QtCore, QtGui  # type: ignore

from .ui_setup import apply_ui, Services
from src.modules.utils.telemetry import log_event, log_exception, init_telemetry
from src.modules import generation as gen_mod
try:
    from src.modules.config_store import load_gui_config, save_gui_config
except Exception:
    load_gui_config = lambda: {}
    def save_gui_config(cfg): pass

print("Loaded generation.py from:", gen_mod.__file__)
print("Generation ready sentinel:", getattr(gen_mod, "GENERATION_MODULE_READY", False))


class MainWindow(QtW.QMainWindow):
    def __init__(self, services: Optional[Services] = None, parent: Optional[QtW.QWidget] = None):
        super().__init__(parent)
        init_telemetry()
        self.services = services or Services()
        self.setWindowTitle("SDXL Cockpit")
        apply_ui(self, self.services)

        # State
        self._dark_mode = False
        self._busy = False
        self._cancel_event: Optional[threading.Event] = None
        self._workers: Set[_Worker] = set()
        self._current_model_kind = "image"
        self._model_ready: bool = False  # NEW: gate Generate until model fully loaded
        # Refine cancellation bookkeeping
        self._refine_cancelled: bool = False
        self._refine_worker: Optional[_Worker] = None
        self._refine_timed_out: bool = False  # NEW

        # Capture BEFORE/AFTER for refiner popup
        self._refine_before_pil = None
        self._refine_before_qimg: Optional[QtGui.QImage] = None

        self._last_batch_pil: List[Any] = []
        self._last_image_pil = None
        self._last_image_qt: Optional[QtGui.QImage] = None
        self._last_video_frames_qt: List[QtGui.QImage] = []
        self._last_video_frames_pil: List[Any] = []
        self._video_timer: Optional[QtCore.QTimer] = None
        self._video_frame_index = 0

        self._output_dir = Path("outputs/images"); self._output_dir.mkdir(parents=True, exist_ok=True)
        self._last_saved_path: Optional[Path] = None

        # Config
        self._cfg = load_gui_config()
        remember_flag = bool(self._cfg.get("remember_output_dir", True))
        cache_flag = bool(self._cfg.get("use_pipeline_disk_cache", True))
        if "remember_save_chk" in self._ui: self._ui["remember_save_chk"].setChecked(remember_flag)
        if "pipeline_cache_chk" in self._ui: self._ui["pipeline_cache_chk"].setChecked(cache_flag)

        # Restore last-used UI values
        self._apply_cfg_to_ui()

        # Apply saved theme preference (dark by default)
        try:
            if bool(self._cfg.get("dark_mode", True)):
                self._apply_dark_palette()
                self.setStyleSheet("")
            else:
                self._apply_light_palette()
                self.setStyleSheet("")
        except Exception as e:
            log_exception(e, context="apply_saved_theme")

        if hasattr(gen_mod, "set_disk_cache_enabled") and hasattr(gen_mod, "is_disk_cache_enabled"):
            try:
                if gen_mod.is_disk_cache_enabled() != cache_flag:
                    gen_mod.set_disk_cache_enabled(cache_flag)
            except Exception as e:
                log_exception(e, context="disk_cache_toggle_init")
        else:
            print("[Diagnostics] generation.set_disk_cache_enabled missing (partial import?).", flush=True)
            log_event({"event": "generation_partial_import", "missing": "set_disk_cache_enabled"})

        if remember_flag:
            cfg_dir = self._cfg.get("last_output_dir")
            if cfg_dir:
                with contextlib.suppress(Exception):
                    self._output_dir = Path(cfg_dir); self._output_dir.mkdir(parents=True, exist_ok=True)

        last_model = self._cfg.get("last_model_path")
        if last_model: os.environ["MODEL_ID_OR_PATH"] = last_model

        # Restore window geometry/state
        with contextlib.suppress(Exception):
            geo = self._cfg.get("window_geometry")
            state = self._cfg.get("window_state")
            if isinstance(geo, list): self.restoreGeometry(bytes(geo))
            if isinstance(state, list): self.restoreState(bytes(state))

        # Gallery
        self._gallery_items: List[dict] = []
        self._favorites_items: List[dict] = []  # NEW: favourites store
        self._gallery_dir = Path(
            self._cfg.get("gallery_dir")
            or os.environ.get("GALLERY_DIR")
            or str(Path.cwd() / "gallery")
        )
        self._gallery_dir.mkdir(parents=True, exist_ok=True)
        auto_g = bool(self._cfg.get("auto_save_to_gallery", True))
        if "auto_gallery_chk" in self._ui: self._ui["auto_gallery_chk"].setChecked(auto_g)
        self._gallery_viewer: Optional[_GalleryViewer] = None

        # Populate models
        self._populate_model_combo()
        self._set_buttons_idle()
        log_event({"event": "ui_init"})

        # Prompt styling callbacks
        if "prompt_edit" in self._ui:
            self._ui["prompt_edit"].textChanged.connect(self._update_generate_action_style)  # type: ignore
        if "negative_edit" in self._ui:
            self._ui["negative_edit"].textChanged.connect(self._update_generate_action_style)  # type: ignore
        QtCore.QTimer.singleShot(0, self._update_generate_action_style)
        QtCore.QTimer.singleShot(250, self._initial_gallery_scan)

        # VRAM telemetry timer (non-blocking tick; paused during heavy jobs)
        self._vram_timer = QtCore.QTimer(self)
        self._vram_timer.timeout.connect(self._vram_telemetry_tick)  # type: ignore
        self._vram_timer.start(20000)
        self._last_vram_tuple = None

        # Populate upscaler models
        self._populate_upscaler_combo()
        self._ui["upscaler_combo"].currentIndexChanged.connect(self._update_upscale_enable)
        self._update_upscale_enable()

        # Initialize Anatomy Guard from checkbox default
        chk = self._ui.get("anatomy_guard_chk")
        if chk:
            os.environ["ANATOMY_GUARD"] = "1" if chk.isChecked() else "0"

        # Wire persistence handlers and optional VRAM auto-show
        self._wire_persistence_handlers()
        # Avoid xFormers attempt for the refiner unless explicitly enabled
        os.environ.setdefault("SDXL_REFINER_XFORMERS", "0")
        # NOTE: Do NOT preload the refiner here; it competes with first model load.
        # We'll trigger async preload after the model is loaded or after first generation.
        if bool(self._cfg.get("show_vram_on_start", False)):
            with contextlib.suppress(Exception):
                self.on_show_vram()

        # Wire generate/save actions
        act = self._ui.get("act_generate");      btn = self._ui.get("btn_generate")
        if act: act.triggered.connect(self.on_generate)  # type: ignore
        if btn: btn.clicked.connect(self.on_generate)    # type: ignore

        actv = self._ui.get("act_generate_video"); btnv = self._ui.get("btn_generate_video")
        if actv: actv.triggered.connect(self.on_generate_video)  # type: ignore
        if btnv: btnv.clicked.connect(self.on_generate_video)    # type: ignore

        save = self._ui.get("act_save_image"); save_as = self._ui.get("act_save_image_as")
        if save: save.triggered.connect(self.on_save_image)        # type: ignore
        if save_as: save_as.triggered.connect(self.on_save_image_as)  # type: ignore

        savev = self._ui.get("act_save_video"); savev_as = self._ui.get("act_save_video_as")
        if savev: savev.triggered.connect(self.on_save_video)      # type: ignore
        if savev_as: savev_as.triggered.connect(self.on_save_video_as)  # type: ignore

        # Also wire the toolbar buttons (if present)
        btn_save = self._ui.get("btn_save_image"); btn_save_as = self._ui.get("btn_save_image_as")
        if btn_save: btn_save.clicked.connect(self.on_save_image)          # type: ignore
        if btn_save_as: btn_save_as.clicked.connect(self.on_save_image_as) # type: ignore
        btn_sv = self._ui.get("btn_save_video"); btn_sv_as = self._ui.get("btn_save_video_as")
        if btn_sv: btn_sv.clicked.connect(self.on_save_video)              # type: ignore
        if btn_sv_as: btn_sv_as.clicked.connect(self.on_save_video_as)     # type: ignore

        # Wire cancel + show VRAM
        actc = self._ui.get("act_cancel"); btnc = self._ui.get("btn_cancel")
        if actc: actc.triggered.connect(self.on_cancel_generate)  # type: ignore
        if btnc: btnc.clicked.connect(self.on_cancel_generate)    # type: ignore

        actvram = self._ui.get("act_show_vram"); btnvram = self._ui.get("btn_show_vram")
        if actvram: actvram.triggered.connect(self.on_show_vram)  # type: ignore
        if btnvram: btnvram.clicked.connect(self.on_show_vram)    # type: ignore

        # Apply saved theme preference (dark by default if not set)
        try:
            if bool(self._cfg.get("dark_mode", True)):
                self._apply_dark_palette()
                self.setStyleSheet("")
            else:
                self._apply_light_palette()
                self.setStyleSheet("")
        except Exception as e:
            log_exception(e, context="apply_saved_theme")
            self.setStyleSheet("")

        # --- Inpaint editor wiring (stack + fullscreen) ---
        self._ensure_central_stack()
        self._install_inpaint_action()

        # --- Star button (favourites) wiring ---
        self._ensure_star_button()

    # -------- Persistence helpers --------
    def _apply_cfg_to_ui(self):
        ui = self._ui
        def set_if(name, val, fn):
            if name in ui:
                fn(ui[name], val)

        set_if("prompt_edit", self._cfg.get("last_prompt",""), lambda w,v: w.setPlainText(v))
        set_if("negative_edit", self._cfg.get("last_negative",""), lambda w,v: w.setPlainText(v))
        set_if("width_spin", int(self._cfg.get("last_width",1280)), lambda w,v: w.setValue(v))
        set_if("height_spin", int(self._cfg.get("last_height",720)), lambda w,v: w.setValue(v))
        set_if("steps_spin", int(self._cfg.get("last_steps",30)), lambda w,v: w.setValue(v))
        set_if("cfg_spin", float(self._cfg.get("last_cfg",7.5)), lambda w,v: w.setValue(v))
        set_if("seed_spin", int(self._cfg.get("last_seed",0)), lambda w,v: w.setValue(v))
        set_if("batch_spin", int(self._cfg.get("last_batch",1)), lambda w,v: w.setValue(v))
        set_if("seed_auto_chk", bool(self._cfg.get("seed_auto_enabled", False)), lambda w,v: w.setChecked(v))
        set_if("anatomy_guard_chk", bool(self._cfg.get("anatomy_guard_enabled", True)), lambda w,v: w.setChecked(v))
        set_if("auto_refine_chk", bool(self._cfg.get("auto_refine_enabled", False)), lambda w,v: w.setChecked(v))

        if "sampler_combo" in ui:
            want = str(self._cfg.get("last_sampler","euler_a"))
            idx = ui["sampler_combo"].findText(want)
            if idx >= 0: ui["sampler_combo"].setCurrentIndex(idx)

        if "precision_dropdown" in ui:
            want = str(self._cfg.get("precision_mode","FP32"))
            idx = ui["precision_dropdown"].findText(want)
            if idx >= 0: ui["precision_dropdown"].setCurrentIndex(idx)

        # Default Turbo decode to FP16 (faster VAE decode for Turbo models)
        if "turbo_decode_dropdown" in ui:
            want = str(self._cfg.get("turbo_decode_mode","FP16"))
            idx = ui["turbo_decode_dropdown"].findText(want)
            if idx >= 0: ui["turbo_decode_dropdown"].setCurrentIndex(idx)
    def _wire_persistence_handlers(self):
        def save():
            with contextlib.suppress(Exception): save_gui_config(self._cfg)

        # Text edits
        if "prompt_edit" in self._ui:
            self._ui["prompt_edit"].textChanged.connect(lambda:
                (self._cfg.__setitem__("last_prompt", self._ui["prompt_edit"].toPlainText()), save()))  # type: ignore
        if "negative_edit" in self._ui:
            self._ui["negative_edit"].textChanged.connect(lambda:
                (self._cfg.__setitem__("last_negative", self._ui["negative_edit"].toPlainText()), save()))  # type: ignore

        # Spin boxes
        def bind_spin(name, key, cast=int):
            if name in self._ui:
                self._ui[name].valueChanged.connect(lambda v:
                    (self._cfg.__setitem__(key, cast(v)), save()))  # type: ignore
        bind_spin("width_spin", "last_width", int)
        bind_spin("height_spin", "last_height", int)
        bind_spin("steps_spin", "last_steps", int)
        bind_spin("seed_spin", "last_seed", int)
        bind_spin("batch_spin", "last_batch", int)
        bind_spin("cfg_spin", "last_cfg", float)

        # Checkboxes
        def bind_chk(name, key):
            if name in self._ui:
                self._ui[name].toggled.connect(lambda b:
                    (self._cfg.__setitem__(key, bool(b)), save()))  # type: ignore
        bind_chk("seed_auto_chk", "seed_auto_enabled")
        bind_chk("anatomy_guard_chk", "anatomy_guard_enabled")
        bind_chk("auto_refine_chk", "auto_refine_enabled")

        # Combos
        if "sampler_combo" in self._ui:
            self._ui["sampler_combo"].currentTextChanged.connect(lambda s:
                (self._cfg.__setitem__("last_sampler", str(s)), save()))  # type: ignore
        if "precision_dropdown" in self._ui:
            self._ui["precision_dropdown"].currentTextChanged.connect(lambda s:
                (self._cfg.__setitem__("precision_mode", str(s)), save()))  # type: ignore
        if "turbo_decode_dropdown" in self._ui:
            self._ui["turbo_decode_dropdown"].currentTextChanged.connect(lambda s:
                (self._cfg.__setitem__("turbo_decode_mode", str(s)), save()))  # type: ignore

    # --- Anatomy Guard handlers (additive) ---
    def on_toggle_anatomy_guard(self, checked: bool):
        os.environ["ANATOMY_GUARD"] = "1" if checked else "0"
        log_event({"event":"anatomy_guard_toggle","enabled":bool(checked)})
        self._status(f"Anatomy Guard {'ON' if checked else 'OFF'}")

    def on_apply_anatomy_guard(self):
        os.environ.update({
            "ANATOMY_GUARD": "1",
            "ANATOMY_GUARD_TURBO_CFG_MAX": "5.5",
            "ANATOMY_GUARD_TURBO_STEPS": "10",
            "ANATOMY_GUARD_SAMPLER": "dpmpp_2m_karras",
            "ANATOMY_GUARD_GUIDANCE_RESCALE": "0.7",
            "ANATOMY_GUARD_POSITIVE_AUG": "1",
        })
        log_event({"event":"anatomy_guard_apply_defaults"})
        self._status("Anatomy Guard defaults applied.")

    # ---------- Model handling ----------
    def _populate_model_combo(self):
        combo = self._ui.get("model_combo")
        if not combo: return
        combo.clear()
        try:
            entries = gen_mod.list_all_models()
        except Exception:
            entries = []
        if os.getenv("REFRESH_MODEL_LIST","0")=="1":
            from importlib import reload
            with contextlib.suppress(Exception):
                reload(gen_mod)
                entries = gen_mod.list_all_models()
        for entry in entries:
            if isinstance(entry, dict) and "label" in entry:
                combo.addItem(entry["label"], entry)
        has_sdxl = any("stable-diffusion-xl-base" in (combo.itemData(i) or {}).get("path","")
                       for i in range(combo.count()))
        if not has_sdxl:
            combo.addItem("HF: SDXL Base (hub)", {
                "label":"HF: SDXL Base (hub)",
                "path":"stabilityai/stable-diffusion-xl-base-1.0",
                "kind":"image"
            });
        current_target = None
        with contextlib.suppress(Exception):
            current_target = gen_mod.current_model_target()

        # Prefer persisted selection first
        selected = False
        if current_target:
            for i in range(combo.count()):
                data = combo.itemData(i)
                if data and data.get("path")==current_target:
                    combo.setCurrentIndex(i); selected = True; break

        # If nothing persisted, default to DreamShaper XL v2 Turbo
        if not selected and not self._cfg.get("last_model_path"):
            dream_key = "dreamshaper-xl-v2-turbo"
            for i in range(combo.count()):
                data = combo.itemData(i) or {}
                path = str(data.get("path","")).lower()
                if dream_key in path:
                    combo.setCurrentIndex(i)
                    self._current_model_kind = data.get("kind","image")
                    selected = True
                    break

        self._update_model_kind_enable()

    # 1) Keep on_load_model as a top-level function (module helper)
    def on_load_model(self):
        # Prevent concurrent loads – overlapping loads slow/hang force_load_pipeline
        if self._busy:
            self._status("Busy... please wait."); return
        combo = self._ui.get("model_combo")
        if not combo:
            self._status("Model combo missing."); return
        data = combo.currentData()
        if not data:
            self._status("No model selected."); return
        kind = data.get("kind","image")
        path = data.get("path")
        if not path:
            self._status("Model path missing."); return
        self._current_model_kind = kind
        self._model_ready = False  # mark not ready during load
        self._update_model_kind_enable()
        self._update_generate_action_style()
        self._status("Scheduling model load...")
        self._busy = True
        self._set_buttons_running()
        self.repaint()
        sampler = self._ui.get("sampler_combo").currentText() if "sampler_combo" in self._ui else "euler_a"
        parent = self

        class _ModelLoader(_Worker):
            def __init__(self): super().__init__(fn=self._do, args=(), parent=parent)
            def _do(self, progress_cb):
                try:
                    with contextlib.suppress(Exception):
                        progress_cb("Releasing pipeline VRAM...")
                        gen_mod.release_pipeline(free_ram=False)
                    progress_cb("Setting target...")
                    gen_mod.set_model_target(path)
                    os.environ["MODEL_ID_OR_PATH"] = path
                    progress_cb("Loading pipeline (async)...")
                    gen_mod.force_load_pipeline(sampler=sampler, device="cuda")
                    return True
                except Exception as ex:
                    log_exception(ex, context="async_model_load_internal"); raise

        worker = _ModelLoader()
        worker.progress.connect(lambda m: self._status(m))  # type: ignore

        def _done(_):
            self._reset_preview()
            log_event({"event":"model_loaded","model":path,"kind":kind})
            label = data.get('label', path)
            self._status(f"Model loaded: {label}")
            self._cfg["last_model_path"] = path
            with contextlib.suppress(Exception): save_gui_config(self._cfg)
            self._model_ready = True
            self._status_highlight(f"{label} READY FOR GENERATION")
            with contextlib.suppress(Exception):
                QtCore.QTimer.singleShot(0, lambda: getattr(gen_mod, "trigger_background_prefetch", lambda: None)())
            self._busy = False
            self._set_buttons_idle()
            self._update_model_kind_enable()
            self._update_generate_action_style()
            self._vram_telemetry_tick()

        def _err(e: Exception):
            log_exception(e, "async_model_load")
            self._status(f"Load failed: {e}")
            self._model_ready = False
            self._busy = False
            self._set_buttons_idle()

        worker.result.connect(_done)   # type: ignore
        worker.error.connect(_err)     # type: ignore
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))  # type: ignore
        log_event({"event":"model_loader_spawn","path":path,"sampler":sampler})
        worker.started.connect(lambda: log_event({"event":"model_loader_thread_started"}))  # type: ignore
        self._workers.add(worker); worker.start()

    # ---------- Gallery ----------
    def _init_gallery(self):
        if "gallery_widget" in self._ui and "fav_widget" in self._ui:
            return
        dock = self._ui.get("gallery_dock")
        tabs = self._ui.get("gallery_tabs")
        if not dock:
            dock = QtW.QDockWidget("Gallery", self)
            self._ui["gallery_dock"] = dock
        if not tabs:
            tabs = QtW.QTabWidget()
            self._ui["gallery_tabs"] = tabs

        # Main gallery list
        if "gallery_widget" not in self._ui:
            lst = QtW.QListWidget()
            lst.setIconSize(QtCore.QSize(96,96))
            lst.itemClicked.connect(self._on_gallery_item_clicked)                # type: ignore
            lst.itemDoubleClicked.connect(self._on_gallery_item_double_clicked)   # type: ignore
            self._ui["gallery_widget"] = lst
            tabs.addTab(lst, "Gallery")

        # Favourites list (NEW)
        if "fav_widget" not in self._ui:
            fav = QtW.QListWidget()
            fav.setIconSize(QtCore.QSize(96,96))
            fav.itemClicked.connect(self._on_fav_item_clicked)                    # type: ignore
            fav.itemDoubleClicked.connect(self._on_fav_item_double_clicked)       # type: ignore
            self._ui["fav_widget"] = fav
            tabs.addTab(fav, "Favourites")

        dock.setWidget(tabs)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._gallery_viewer = None

    def _on_gallery_item_clicked(self, item):
        try:
            lst=self._ui.get("gallery_widget")
            if not lst: return
            idx = lst.row(item)
            if 0<=idx<len(self._gallery_items):
                entry = self._gallery_items[idx]
                self._last_image_pil=entry["pil"]
                self._last_image_qt=entry["qimg"]
                paired = entry.get("paired_qimg")
                if paired is not None:
                    with contextlib.suppress(Exception):
                        comp = self._make_side_by_side_qimage(entry["qimg"], paired)
                        self._display_qimage(comp)
                        return
                self._display_qimage(self._last_image_qt)
                if "btn_refine" in self._ui: self._ui["btn_refine"].setEnabled(True)
        except Exception as e:
            log_exception(e,"gallery_click")

    def _on_gallery_item_double_clicked(self, item):
        try:
            lst=self._ui.get("gallery_widget")
            if not lst: return
            idx = lst.row(item)
            if 0 <= idx < len(self._gallery_items):
                self._open_gallery_viewer(idx)
        except Exception as e:
            log_exception(e, context="gallery_double_click")

    def _open_gallery_viewer(self, index: int):
        if not self._gallery_items:
            return
        if self._gallery_viewer is None:
            self._gallery_viewer = _GalleryViewer(self)
        self._gallery_viewer.set_items(self._gallery_items, index)
        self._gallery_viewer.show()
        self._gallery_viewer.raise_()
        self._gallery_viewer.activateWindow()

    def _add_to_gallery(self, pil_img):
        try:
            self._init_gallery()
            qimg = self._pil_to_qimage(pil_img)
            icon = QtGui.QIcon(QtGui.QPixmap.fromImage(
                qimg.scaled(96,96,
                            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation)))
            item = QtW.QListWidgetItem(icon, f"{len(self._gallery_items)+1}")
            self._ui["gallery_widget"].addItem(item)
            self._gallery_items.append({"pil": pil_img, "qimg": qimg, "item": item})
        except Exception as e:
            log_exception(e, context="gallery_add")

    # ---------- Startup gallery scan ----------
    def _initial_gallery_scan(self):
        if not self._cfg.get("auto_save_to_gallery", True):
            return
        try:
            from glob import glob
            patterns = ("*.png","*.jpg","*.jpeg","*.webp")
            files: list[str] = []
            for pat in patterns:
                files.extend(glob(str(self._gallery_dir / pat)))
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            files = files[:50]
            if not files:
                return
            self._init_gallery()
            from PIL import Image
            for f in reversed(files):
                with contextlib.suppress(Exception):
                    im = Image.open(f).convert("RGB")
                    self._add_to_gallery(im)
            log_event({"event":"gallery_scan","count":len(files)})
        except Exception as e:
            log_exception(e, context="gallery_scan")

    # ---------- VRAM telemetry ----------
    def _vram_telemetry_tick(self):
        # Skip telemetry while a long GPU job is running to keep UI responsive
        if getattr(self, "_busy", False):
            return
        try:
            import importlib
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                return
            # Do NOT call torch.cuda.synchronize() on the UI thread.
            alloc = int(torch.cuda.memory_allocated() // 1024**2)
            reserved = int(torch.cuda.memory_reserved() // 1024**2)
            total = int(torch.cuda.get_device_properties(0).total_memory // 1024**2)
            has_pipe = False
            try:
                has_pipe = gen_mod.has_pipeline() if hasattr(gen_mod,"has_pipeline") else getattr(gen_mod,"_PIPELINE",None) is not None
            except Exception:
                has_pipe = False
            tup = (alloc,reserved,has_pipe,self._current_model_kind)
            if tup != getattr(self, "_last_vram_tuple", None) or os.environ.get("VRAM_SNAPSHOT_DEDUP","1") == "0":
                log_event({
                    "event":"vram_snapshot",
                    "alloc_mb":alloc,
                    "reserved_mb":reserved,
                    "total_mb":total,
                    "model_kind":self._current_model_kind,
                    "has_pipeline": has_pipe
                })
                self._last_vram_tuple = tup
        except Exception:
            pass

    # Helpers to pause/resume VRAM timer around long jobs
    def _pause_vram_timer(self):
        try:
            if getattr(self, "_vram_timer", None):
                self._vram_timer.stop()
        except Exception:
            pass

    def _resume_vram_timer(self):
        try:
            if getattr(self, "_vram_timer", None):
                self._vram_timer.start(20000)  # 20s
        except Exception:
            pass

    # ---------- Theme toggle ----------
    def on_toggle_theme(self):
        """
        Toggle between dark and light palettes, independent of QSS blue accents.
        Persists 'dark_mode' to config.
        """
        try:
            dark_now = bool(self._cfg.get("dark_mode", True))
            if dark_now:
                self._apply_light_palette()
                self.setStyleSheet("")  # clear any QSS forcing colors
                self._cfg["dark_mode"] = False
                self._status("Light theme enabled.")
            else:
                self._apply_dark_palette()
                # Optional: keep QSS if it only styles widgets, but avoid blue accents
                # Clear QSS to rely purely on palette
                self.setStyleSheet("")
                self._cfg["dark_mode"] = True
                self._status("Dark theme enabled.")
            with contextlib.suppress(Exception):
                from src.modules.config_store import save_gui_config
                save_gui_config(self._cfg)
        except Exception as e:
            log_exception(e, context="toggle_theme")
            self._status(f"Theme toggle failed: {e}")

    def _apply_dark_palette(self):
        """
        Graphite dark palette (no blue). Works with Fusion style.
        """
        try:
            app = QtW.QApplication.instance()
            if app is None:
                return
            app.setStyle("Fusion")
            p = QtGui.QPalette()
            base = QtGui.QColor(35, 35, 38)
            panel = QtGui.QColor(45, 45, 48)
            text = QtGui.QColor(220, 220, 220)
            disabled = QtGui.QColor(140, 140, 140)
            highlight = QtGui.QColor(80, 80, 80)  # neutral highlight, not blue

            p.setColor(QtGui.QPalette.ColorRole.Window, panel)
            p.setColor(QtGui.QPalette.ColorRole.WindowText, text)
            p.setColor(QtGui.QPalette.ColorRole.Base, base)
            p.setColor(QtGui.QPalette.ColorRole.AlternateBase, panel)
            p.setColor(QtGui.QPalette.ColorRole.ToolTipBase, panel)
            p.setColor(QtGui.QPalette.ColorRole.ToolTipText, text)
            p.setColor(QtGui.QPalette.ColorRole.Text, text)
            p.setColor(QtGui.QPalette.ColorRole.Button, panel)
            p.setColor(QtGui.QPalette.ColorRole.ButtonText, text)
            p.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
            p.setColor(QtGui.QPalette.ColorRole.Highlight, highlight)
            p.setColor(QtGui.QPalette.ColorRole.HighlightedText, text)

            # Disabled variants
            p.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, disabled)
            p.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, disabled)
            p.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.WindowText, disabled)
            p.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.HighlightedText, disabled)
            p.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(70,70,70))

            app.setPalette(p)
        except Exception as e:
            log_exception(e, context="apply_dark_palette")

    def _apply_light_palette(self):
        """
        Reset to default light palette (no forced dark/blue).
        """
        try:
            app = QtW.QApplication.instance()
            if app is None:
                return
            app.setStyle("Fusion")
            app.setPalette(app.style().standardPalette())
        except Exception as e:
            log_exception(e, context="apply_light_palette")
    # ---------- Busy indicator (non-blocking) ----------
    def _start_busy_indicator(self, label: str):
        self._busy_label = label
        self._busy_started = time.monotonic()
        if not hasattr(self, "_busy_timer") or self._busy_timer is None:
            self._busy_timer = QtCore.QTimer(self)
            self._busy_timer.timeout.connect(self._update_busy_heartbeat)  # type: ignore
        self._busy_timer.start(1000)
        self._update_busy_heartbeat()

    def _update_busy_heartbeat(self):
        try:
            if not getattr(self, "_busy_label", None):
                return
            elapsed = int(time.monotonic() - getattr(self, "_busy_started", time.monotonic()))
            self._status(f"{self._busy_label}... ({elapsed}s)")
        except Exception:
            pass

    def _stop_busy_indicator(self):
        with contextlib.suppress(Exception):
            if getattr(self, "_busy_timer", None):
                self._busy_timer.stop()
        self._busy_label = None
        self._busy_started = None
        with contextlib.suppress(Exception):
            QtW.QApplication.restoreOverrideCursor()

    def _on_busy_progress(self, msg: str):
        # Suppress noisy progress if refine was cancelled
        if getattr(self, "_refine_cancelled", False):
            return
        if msg:
            self._status(str(msg))

    # ---------- Cancel ----------
    def on_cancel_generate(self):
        if self._busy and self._cancel_event and not self._cancel_event.is_set():
            self._cancel_event.set()
            self._status("Cancelling...")
    # ---------- Refine ----------
    def on_refine(self):
        if self._current_model_kind == "video":
            self._status("Refine not supported for video."); return
        if self._last_image_pil is None:
            self._status("No image to refine."); return
        if self._busy:
            self._status("Busy... please wait."); return
        # Stop blinking once user initiates refine; keep enabled
        if hasattr(self, "_stop_refine_attention"):
            self._stop_refine_attention(disable=False)
        # Capture BEFORE copy for popup/preview pairing
        self._refine_before_pil = self._last_image_pil
        self._refine_before_qimg = self._last_image_qt

        # Background refine to avoid freezing UI while refiner loads/executes
        self._busy = True
        self._refine_cancelled = False
        self._set_buttons_running()
        self._pause_vram_timer()
        self._start_busy_indicator("Refining")
        log_event({"event": "refine_start"})
        worker = _Worker(fn=self._do_refine, args=(), parent=self)
        self._refine_worker = worker
        worker.progress.connect(self._on_busy_progress)               # type: ignore
        worker.result.connect(self._on_refine_done)      # type: ignore
        worker.error.connect(self._on_refine_error)      # type: ignore
        worker.finished.connect(lambda w=worker: (self._cleanup_worker(w), setattr(self, "_refine_worker", None)))  # type: ignore
        self._workers.add(worker); worker.start()

        # Optional safety timeout (default 45s)
        try:
            tout = int(os.environ.get("REFINE_TIMEOUT_S", "45"))
            if tout > 0:
                self._refine_timeout = QtCore.QTimer(self)
                self._refine_timeout.setSingleShot(True)
                self._refine_timeout.timeout.connect(self._on_refine_timeout)  # type: ignore
                self._refine_timeout.start(tout * 1000)
        except Exception:
            pass

    def _on_refine_timeout(self):
        # If still refining, stop heartbeat and mute further progress
        if getattr(self, "_busy_label", None) == "Refining" and not getattr(self, "_refine_cancelled", False):
            self._refine_timed_out = True  # NEW: mark timed out but not cancelled
            try:
                if self._refine_worker:
                    self._refine_worker.progress.disconnect(self._on_busy_progress)  # type: ignore
            except Exception:
                pass
            self._stop_busy_indicator()
            self._status("Refine timed out. Waiting for worker to finish...")

    def _do_refine(self, progress_cb):
        # Free GPU VRAM so the refiner can run fast
        with contextlib.suppress(Exception):
            progress_cb("Releasing GPU VRAM for refiner...")
            gen_mod.release_pipeline(free_ram=False)

        # Preload/ensure refiner
        with contextlib.suppress(Exception):
            from src.modules import refiner_module as _ref_pre
            progress_cb("Loading refiner...")
            _ref_pre.ensure_loaded_async()

        from importlib import import_module
        progress_cb("Preparing refiner...")
        ref_mod = import_module("src.modules.refiner_module")
        fn = getattr(ref_mod, "refine_image_from_rgb", None)
        if not fn:
            raise RuntimeError("Refiner unavailable")
        image = self._last_image_pil
        prompt = self._ui["prompt_edit"].toPlainText().strip()
        negative = self._ui["negative_edit"].toPlainText().strip() or None
        # Faster default (env overrides still honored)
        steps = int(os.environ.get("SDXL_REFINER_STEPS", "6"))
        strength = float(os.environ.get("SDXL_REFINER_STRENGTH", "0.35"))
        progress_cb("Running refiner...")
        refined = fn(image, prompt=prompt, negative_prompt=negative, steps=steps, strength=strength)
        if refined is None:
            raise RuntimeError("Refine failed")
        progress_cb("Finalizing...")
        return refined

    def _on_refine_done(self, refined_img):
        try:
            # Accept result even if it arrived after timeout
            if getattr(self, "_refine_timed_out", False):
                log_event({"event": "refine_finished_after_timeout"})
            # Build refined qimage and show popup side-by-side with BEFORE
            refined_qimg = self._pil_to_qimage(refined_img)
            before_qimg = self._refine_before_qimg
            self._last_image_pil = refined_img
            self._last_image_qt = refined_qimg
            self._display_qimage(self._last_image_qt)
            self._status("Refined." if not getattr(self, "_refine_timed_out", False) else "Refined (finished after timeout).")
            log_event({"event": "refine_success"})
            with contextlib.suppress(Exception):
                self._add_to_gallery(refined_img)
                if before_qimg is not None and self._gallery_items:
                    refined_idx = len(self._gallery_items) - 1
                    orig_idx = next((i for i, it in enumerate(self._gallery_items) if it.get("qimg") is before_qimg), None)
                    if orig_idx is not None:
                        self._gallery_items[orig_idx]["paired_qimg"] = refined_qimg
                        self._gallery_items[refined_idx]["paired_qimg"] = before_qimg
            if before_qimg is not None:
                with contextlib.suppress(Exception):
                    self._show_before_after_popup(before_qimg, refined_qimg)

            # Post-refine housekeeping: free some VRAM; do NOT force-load pipeline on UI thread
            with contextlib.suppress(Exception):
                from src.modules import refiner_module as _ref
                for name in ("release", "unload", "to_cpu"):
                    fn = getattr(_ref, name, None)
                    if callable(fn):
                        fn()
                        break
            with contextlib.suppress(Exception):
                import torch
                torch.cuda.empty_cache()

            # Removed: blocking warm-up that froze UI
            # with contextlib.suppress(Exception):
            #     sampler = self._ui["sampler_combo"].currentText() if "sampler_combo" in self._ui : "euler_a"
            #     QtCore.QTimer.singleShot(0, lambda s=sampler: gen_mod.force_load_pipeline(sampler=s, device="cuda"))
        finally:
            self._busy = False
            self._stop_busy_indicator()
            if hasattr(self, "_stop_refine_attention"):
                self._stop_refine_attention(disable=False)
            with contextlib.suppress(Exception):
                if hasattr(self, "_refine_timeout") and self._refine_timeout:
                    self._refine_timeout.stop(); self._refine_timeout.deleteLater(); self._refine_timeout=None
            self._refine_timed_out = False
            self._refine_cancelled = False
            self._set_buttons_idle()
            self._update_model_kind_enable()
            self._resume_vram_timer()

    def _on_refine_error(self, err: Exception):
        log_exception(err, "refine_worker")
        self._status(f"Refine failed: {err}")
        self._busy = False
        self._stop_busy_indicator()
        # Stop refine attention (no blink) and keep button state managed by _update_model_kind_enable
        if hasattr(self, "_stop_refine_attention"):
            self._stop_refine_attention(disable=False)
        self._set_buttons_idle()
        self._update_model_kind_enable()
        self._resume_vram_timer()

    # ---------- Image generation ----------
    def on_generate(self):
        # Model must be ready and in image mode
        if not getattr(self, "_model_ready", False):
            self._status("Model not ready yet. Please wait for the READY message.")
            return
        if self._busy or self._current_model_kind != "image":
            return
        # Stop attention
        if hasattr(self, "_stop_refine_attention"):
            self._stop_refine_attention(disable=True)
        ui = self._ui
        prompt = ui["prompt_edit"].toPlainText().strip()
        if not prompt:
            self._status_highlight("PLEASE ENTER A PROMPT AND NEGATIVE PROMPT\nHint (negative): blurry, low quality, watermark, extra fingers, extra limbs, text")
            return
        negative = ui["negative_edit"].toPlainText().strip()
        steps = ui["steps_spin"].value()
        cfg = ui["cfg_spin"].value()
        w = ui["width_spin"].value()
        h = ui["height_spin"].value()
        seed = ui["seed_spin"].value()
        batch = ui["batch_spin"].value() if "batch_spin" in ui else 1
        sampler = ui["sampler_combo"].currentText() if "sampler_combo" in ui else "euler_a"
        # Auto-randomize seed when enabled
        try:
            auto_chk = ui.get("seed_auto_chk")
            if auto_chk and auto_chk.isChecked():
                import secrets
                new_seed = int(secrets.randbits(31)) or 1
                seed = new_seed
                if "seed_spin" in ui:
                    ui["seed_spin"].setValue(new_seed)
                self._cfg["last_seed"] = int(new_seed)
                with contextlib.suppress(Exception):
                    save_gui_config(self._cfg)
        except Exception:
            pass
        self._busy = True
        self._cancel_event = threading.Event()
        self._set_buttons_running()
        self._pause_vram_timer()
        self._status("Generating...")
        args = dict(prompt=prompt, negative=negative, steps=steps, cfg=cfg,
                    w=w, h=h, seed=seed, batch=batch, sampler=sampler)

        worker = _Worker(fn=self._do_generate_image, args=(args,), parent=self)
        worker.progress.connect(self._on_generate_progress)  # type: ignore
        worker.result.connect(self._on_generate_done)        # type: ignore
        worker.error.connect(self._on_generate_error)        # type: ignore
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))  # type: ignore
        self._workers.add(worker); worker.start()

    def _do_generate_image(self, args: dict, progress_cb):
        prompt = args["prompt"]; negative = args["negative"]; steps = args["steps"]; cfg = args["cfg"]
        w = args["w"]; h = args["h"]; seed = args["seed"]; batch = args["batch"]; sampler = args["sampler"]
        cancel = self._cancel_event
        progress_cb("Loading pipeline...")
        if os.environ.get("SHOW_CACHE_STATUS", "1") == "1":
            with contextlib.suppress(Exception):
                from src.modules.generation import disk_cache_root
                cache_root = disk_cache_root()
                cur = gen_mod.current_model_target()
                if cache_root.replace("\\", "/") in cur.replace("\\", "/"):
                    self._status("Using disk-cached pipeline.")
        try:
            imgs_pil = gen_mod.generate_images(
                prompt=prompt, negative=negative, steps=steps, cfg=cfg,
                width=w, height=h, seed=seed, batch=batch, sampler=sampler,
                device="cuda", progress_cb=progress_cb, cancel_event=cancel
            )
        except RuntimeError as e:
            if str(e).lower() == "cancelled":
                return ([], [])
            if "cuda" in str(e).lower():
                progress_cb("Retrying on CPU...")
                imgs_pil = gen_mod.generate_images(
                    prompt=prompt, negative=negative, steps=steps, cfg=cfg,
                    width=w, height=h, seed=seed, batch=batch, sampler=sampler,
                    device="cpu", progress_cb=progress_cb, cancel_event=cancel
                )
            else:
                raise
        qimg_list = []
        for im in imgs_pil:
            with contextlib.suppress(Exception):
                qimg_list.append(gen_mod.pil_to_qimage(im))  # fixed name
        log_event({"event": "generation", "count": len(qimg_list), "steps": steps, "cfg": cfg, "w": w, "h": h, "sampler": sampler})
        progress_cb("Finalizing")
        return (imgs_pil, qimg_list)

    def _on_generate_progress(self, msg: str):
        if os.environ.get("VERBOSE_TELEMETRY") == "1":
            log_event({"event": "progress", "message": msg})
        self._status(msg)

    def _on_generate_done(self, result):
        try:
            pil_list, qimg_list = result if isinstance(result, tuple) else ([], result or [])
            if not qimg_list:
                if self._cancel_event and self._cancel_event.is_set():
                    self._status("Generation cancelled.")
                else:
                    self._status("No images generated.")
            else:
                self._last_batch_pil = pil_list
                self._last_image_pil = pil_list[0] if pil_list else None
                self._last_image_qt = qimg_list[0]
                self._display_qimage(self._last_image_qt)
                with contextlib.suppress(Exception):
                    for im in pil_list:
                        self._add_to_gallery(im)
                    if "gallery_dock" in self._ui:
                        self._ui["gallery_dock"].show()
                if "btn_refine" in self._ui:
                    self._ui["btn_refine"].setEnabled(True)
                # Start Refine attention (blink green) after generation completes
                if hasattr(self, "_start_refine_attention"):
                    self._start_refine_attention()
                self._status(f"Generated {len(qimg_list)} image(s).")
                self._maybe_auto_save_images(pil_list)

                # IMPORTANT: do NOT preload the refiner here to avoid GPU contention.
                # It will load on-demand when the user clicks Refine (see _do_refine).
                with contextlib.suppress(Exception):
                    log_event({"event": "refiner_preload_skipped"})

                # Optional auto-refine still honored; this will load on demand.
                if bool(self._cfg.get("auto_refine_enabled", False)):
                    QtCore.QTimer.singleShot(0, self.on_refine)
        finally:
            self._busy = False
            self._set_buttons_idle()
            self._update_model_kind_enable()
            self._update_upscaler_combo()
            self._maybe_auto_release_after_job()
            self._vram_telemetry_tick()
            self._resume_vram_timer()

    def _on_generate_error(self, err: Exception):
        if str(err).lower() == "cancelled":
            log_event({"event": ("video_cancelled" if self._current_model_kind == "video" else "image_cancelled")})
            self._status("Cancelled.")
        else:
            log_exception(err, "generate_worker")
            self._status(f"Error: {err}")
        self._busy = False
        self._set_buttons_idle()
        self._update_model_kind_enable()
        self._resume_vram_timer()

    # ---------- Video generation ----------
    def on_generate_video(self):
        if self._busy or self._current_model_kind != "video":
            return
        # Any refine attention should stop if switching to video gen
        if hasattr(self, "_stop_refine_attention"):
            self._stop_refine_attention(disable=True)
        self._busy = True
        self._cancel_event = threading.Event()
        self._set_buttons_running()
        self._pause_vram_timer()
        ui = self._ui
        prompt = ui["prompt_edit"].toPlainText().strip()
        negative = ui["negative_edit"].toPlainText().strip()
        frames = ui["video_frames_spin"].value()
        fps = ui["video_fps_spin"].value()
        w = ui["width_spin"].value()
        h = ui["height_spin"].value()
        steps = ui["steps_spin"].value()
        cfg = ui["cfg_spin"].value()
        seed = ui["seed_spin"].value()

        # Auto-randomize seed when enabled (video)
        try:
            auto_chk = ui.get("seed_auto_chk")
            if auto_chk and auto_chk.isChecked():
                import secrets
                new_seed = int(secrets.randbits(31)) or 1
                seed = new_seed
                if "seed_spin" in ui:
                    ui["seed_spin"].setValue(new_seed)
                self._cfg["last_seed"] = int(new_seed)
                with contextlib.suppress(Exception):
                    save_gui_config(self._cfg)
        except Exception:
            pass

        self._status("Preparing video...")
        log_event({"event": "video_generation_start", "frames": frames, "fps": fps, "w": w, "h": h, "steps": steps, "cfg": cfg})
        worker = _Worker(
            fn=self._do_generate_video,
            args=(prompt, negative, frames, w, h, fps, steps, cfg, seed, self._cancel_event),
            parent=self
        )
        worker.progress.connect(self._on_generate_progress)  # type: ignore
        worker.result.connect(self._on_video_done)           # type: ignore
        worker.error.connect(self._on_generate_error)        # type: ignore
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))  # type: ignore
        self._workers.add(worker); worker.start()

    def _do_generate_video(self, prompt: str, negative: str, frames: int, w: int, h: int,
                           fps: int, steps: int, cfg: float, seed: int, cancel_event, progress_cb):
        frames_pil = gen_mod.generate_video(
            prompt=prompt, negative=negative, frames=frames,
            width=w, height=h, fps=fps, steps=steps, cfg=cfg, seed=seed,
            progress_cb=progress_cb, cancel_event=cancel_event
        )
        qframes = [gen_mod.pil_to_qimage(f) for f in frames_pil]
        progress_cb("Finalizing")
        log_event({"event": "video_generation", "frames": len(qframes), "steps": steps, "cfg": cfg, "w": w, "h": h})
        return (frames_pil, qframes)

    def _on_video_done(self, result):
        pil_frames, q_frames = result if isinstance(result, tuple) else ([], result)
        if not q_frames:
            self._status("Video generation failed.")
        else:
            self._last_video_frames_pil = pil_frames
            self._last_video_frames_qt = q_frames
            self._start_video_preview()
            self._status(f"Generated video ({len(q_frames)} frames).")
            log_event({"event": "video_generation_complete", "frames": len(q_frames)})
            self._maybe_auto_save_video(pil_frames)
        self._busy = False
        self._set_buttons_idle()
        self._update_model_kind_enable()
        self._maybe_auto_release_after_job()
        self._vram_telemetry_tick()
        self._resume_vram_timer()
# ---------- Fullscreen gallery viewer ----------
class _GalleryViewer(QtW.QDialog):
    def __init__(self, parent: QtW.QWidget):
        super().__init__(parent)
        self.setWindowTitle("Gallery Viewer")
        self.setModal(False)
        self.resize(900, 700)
        self._items: List[dict] = []
        self._index = -1
        self._current_qimg: Optional[QtGui.QImage] = None

        self._label = QtW.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("QLabel { background: #202020; color: #DDD; }")
        self._label.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Expanding)

        nav = QtW.QHBoxLayout()
        self._btn_prev = QtW.QPushButton("◀ Prev")
        self._btn_next = QtW.QPushButton("Next ▶")
        self._btn_close = QtW.QPushButton("Close ✕")
        for b in (self._btn_prev, self._btn_next, self._btn_close):
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        nav.addWidget(self._btn_prev); nav.addWidget(self._btn_next); nav.addStretch(1); nav.addWidget(self._btn_close)

        lay = QtW.QVBoxLayout(self)
        lay.addWidget(self._label, 1); lay.addLayout(nav)

        self._btn_prev.clicked.connect(self.prev)    # type: ignore
        self._btn_next.clicked.connect(self.next)    # type: ignore
        self._btn_close.clicked.connect(self.close)  # type: ignore

    def set_items(self, items: List[dict], index: int):
        self._items = items
        self._index = max(0, min(index, len(items) - 1))
        log_event({"event": "gallery_viewer_open", "index": self._index, "count": len(items)})
        self._update_image()

    def prev(self):
        if not self._items: return
        self._index = (self._index - 1) % len(self._items)
        log_event({"event":"gallery_viewer_nav","dir":"prev","index":self._index})
        self._update_image()

    def next(self):
        if not self._items: return
        self._index = (self._index + 1) % len(self._items)
        log_event({"event":"gallery_viewer_nav","dir":"next","index":self._index})
        self._update_image()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_A):
            self.prev(); e.accept(); return
        if e.key() in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_D):
            self.next(); e.accept(); return
        if e.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
            self.close(); e.accept(); return
        super().keyPressEvent(e)

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if e.angleDelta().y() > 0: self.prev()
        else: self.next()
        e.accept()

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e); self._update_scaled()

    def _update_image(self):
        if not self._items or not (0 <= self._index < len(self._items)):
            self._label.setText("No image"); return
        qimg = self._items[self._index].get("qimg")
        if qimg is None:
            self._label.setText("Missing image"); return
        self._current_qimg = qimg
        self._update_scaled()
        self.setWindowTitle(f"Gallery Viewer ({self._index + 1}/{len(self._items)})")

    def _update_scaled(self):
        if self._current_qimg is None: return
        avail = self._label.size()
        pix = QtGui.QPixmap.fromImage(self._current_qimg).scaled(
            avail,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self._label.setPixmap(pix)

    def closeEvent(self, e: QtGui.QCloseEvent):
        with contextlib.suppress(Exception):
            log_event({"event":"gallery_viewer_close","index":self._index})
        super().closeEvent(e)

# ---------- Helper dialog for Before/After refinement ----------
class _BeforeAfterDialog(QtW.QDialog):
    def __init__(self, parent: QtW.QWidget):
        super().__init__(parent)
        self.setWindowTitle("Before ⟷ After")
        self.setModal(False)
        self.resize(1200, 700)
        self._left: Optional[QtGui.QImage] = None
        self._right: Optional[QtGui.QImage] = None

        self._lbl_left = QtW.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self._lbl_right = QtW.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        for lbl, title in ((self._lbl_left, "Before"), (self._lbl_right, "After")):
            lbl.setStyleSheet("QLabel { background:#202020; color:#DDD; }")
            lbl.setMinimumSize(200, 200)
            lbl.setToolTip(title)

        grid = QtW.QGridLayout(self)
        cap_left = QtW.QLabel("Before"); cap_left.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cap_right = QtW.QLabel("After"); cap_right.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        for cap in (cap_left, cap_right):
            cap.setStyleSheet("QLabel { font-weight:bold; }")

        grid.addWidget(cap_left, 0, 0)
        grid.addWidget(cap_right, 0, 1)
        grid.addWidget(self._lbl_left, 1, 0)
        grid.addWidget(self._lbl_right, 1, 1)

    def set_images(self, left: QtGui.QImage, right: QtGui.QImage):
        self._left = left
        self._right = right
        self._update_scaled()

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        self._update_scaled()

    def _update_scaled(self):
        if self._left is None or self._right is None:
            return
        def scaled(img: QtGui.QImage, target: QtW.QLabel) -> QtGui.QPixmap:
            return QtGui.QPixmap.fromImage(img).scaled(
                target.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
        self._lbl_left.setPixmap(scaled(self._left, self._lbl_left))
        self._lbl_right.setPixmap(scaled(self._right, self._lbl_right))

    def closeEvent(self, e: QtGui.QCloseEvent):
        with contextlib.suppress(Exception):
            log_event({"event":"before_after_dialog_close"})
        super().closeEvent(e)

# ---------- Worker Thread Base ----------
try:
    from PySide6 import QtCore as _QtCore  # type: ignore
except Exception:
    try:
        from PyQt5 import QtCore as _QtCore  # type: ignore
    except Exception:
        _QtCore = None

if _QtCore and "_Worker" not in globals():
    SignalType = getattr(_QtCore, "Signal", getattr(_QtCore, "pyqtSignal", None))
    class _Worker(_QtCore.QThread):
        progress = SignalType(str)      # type: ignore
        result = SignalType(object)     # type: ignore
        error = SignalType(Exception)   # type: ignore
        def __init__(self, fn, args=(), kwargs=None, parent=None):
            super().__init__(parent)
            self._fn = fn
            self._args = args
            self._kwargs = kwargs or {}
        def run(self):
            try:
                res = self._fn(*self._args, progress_cb=self._emit_progress, **self._kwargs)
                self.result.emit(res)
            except Exception as e:
                self.error.emit(e)
        def _emit_progress(self, msg: str):
            with contextlib.suppress(Exception):
                self.progress.emit(msg)

# ---------- Helper UI state (module functions to be bound to MainWindow) ----------
def _set_buttons_running(self):
    if "act_generate" in self._ui: self._ui["act_generate"].setEnabled(False)
    if "btn_generate" in self._ui: self._ui["btn_generate"].setEnabled(False)
    if "act_generate_video" in self._ui: self._ui["act_generate_video"].setEnabled(False)
    if "act_cancel" in self._ui: self._ui["act_cancel"].setEnabled(True)
    if "btn_cancel" in self._ui: self._ui["btn_cancel"].setEnabled(True)
    if "model_combo" in self._ui: self._ui["model_combo"].setEnabled(False)
    # Style Cancel red when busy
    if hasattr(self, "_update_cancel_button_style"):
        self._update_cancel_button_style()

def _set_buttons_idle(self):
    if "act_generate" in self._ui:
        self._ui["act_generate"].setEnabled(self._current_model_kind == "image" and not self._busy and getattr(self, "_model_ready", False))
    if "btn_generate" in self._ui:
        self._ui["btn_generate"].setEnabled(self._current_model_kind == "image" and not self._busy and getattr(self, "_model_ready", False))
    if "act_generate_video" in self._ui:
        self._ui["act_generate_video"].setEnabled(self._current_model_kind == "video" and not self._busy and getattr(self, "_model_ready", False))
    if "btn_generate_video" in self._ui:
        self._ui["btn_generate_video"].setEnabled(self._current_model_kind == "video" and not self._busy and getattr(self, "_model_ready", False))
    if "act_cancel" in self._ui: self._ui["act_cancel"].setEnabled(False)
    if "btn_cancel" in self._ui: self._ui["btn_cancel"].setEnabled(False)
    if "model_combo" in self._ui: self._ui["model_combo"].setEnabled(True)
    if hasattr(self, "_update_cancel_button_style"):
        self._update_cancel_button_style()
    if hasattr(self, "_update_generate_action_style"):
        self._update_generate_action_style()

def _get_toolbar_button(self, action_key: str):
    tb = self._ui.get("toolbar_main")
    act = self._ui.get(action_key)
    if not tb or not act:
        return None
    with contextlib.suppress(Exception):
        return tb.widgetForAction(act)
    return None

def _update_cancel_button_style(self):
    # Red while busy (generation/refinement/video), grey (default) otherwise
    red_style = "background:#c62828; color:white; font-weight:bold;"
    off_style = ""
    # Toolbar button for Cancel action
    tbtn = _get_toolbar_button(self, "act_cancel")
    if tbtn:
        tbtn.setStyleSheet(red_style if self._busy else off_style)
    # Optional standalone Cancel QPushButton
    btn = self._ui.get("btn_cancel")
    if btn:
        btn.setStyleSheet(red_style if self._busy else off_style)

# Blink Refine after generation completes
def _toggle_refine_flash(self):
    try:
        self._refine_flash_on = not getattr(self, "_refine_flash_on", False)
        style_on = "background:#2e7d32; color:white; font-weight:bold;"
        style_off = ""

        # Toolbar button for Refine action
        tbtn = _get_toolbar_button(self, "act_refine")
        if tbtn:
            tbtn.setStyleSheet(style_on if self._refine_flash_on else style_off)

        # Optional standalone Refine QPushButton
        btn = self._ui.get("btn_refine")
        if btn:
            btn.setStyleSheet(style_on if self._refine_flash_on else style_off)
    except Exception:
        pass

def _start_refine_attention(self):
    # Only for image mode, when an image exists
    if self._current_model_kind == "video" or self._last_image_pil is None:
        return
    # Ensure Refine is enabled
    if "act_refine" in self._ui:
        self._ui["act_refine"].setEnabled(True)
    if "btn_refine" in self._ui:
        self._ui["btn_refine"].setEnabled(True)
    # Start blinking
    if not hasattr(self, "_refine_blink_timer") or self._refine_blink_timer is None:
        self._refine_blink_timer = QtCore.QTimer(self)
        self._refine_blink_timer.timeout.connect(self._toggle_refine_flash)  # type: ignore
    self._refine_flash_on = False
    self._refine_blink_timer.start(500)
    _toggle_refine_flash(self)  # apply first state immediately

def _stop_refine_attention(self, disable: bool):
    # Stop blinking and clear style; optionally disable the Refine control
    try:
        if hasattr(self, "_refine_blink_timer") and self._refine_blink_timer:
            self._refine_blink_timer.stop()
    except Exception:
        pass
    # Clear styles
    for key, is_action in (("act_refine", True), ("btn_refine", False)):
        if is_action:
            tbtn = _get_toolbar_button(self, key)
            if tbtn:
                tbtn.setStyleSheet("")
        else:
            btn = self._ui.get(key)
            if btn:
                btn.setStyleSheet("")
    if disable:
        if "act_refine" in self._ui:
            self._ui["act_refine"].setEnabled(False)
        if "btn_refine" in self._ui:
            self._ui["btn_refine"].setEnabled(False)

# ---------- UI helpers (status/preview/etc) ----------
def _status(self, msg: str):
    if "status_msg" in self._ui: self._ui["status_msg"].setText(msg)
    log_edit = self._ui.get("log_edit")
    if log_edit and msg and not msg.startswith(("Sampling step","Frame ")):
        log_edit.appendPlainText(msg)
    with contextlib.suppress(Exception):
        if msg:
            log_event({"event":"status","message":msg})

# NEW: highlighted status helper used by on_generate/on_load_model
def _status_highlight(self, msg: str, color: str = "#2e7d32", ms: int = 2000):
    try:
        log_edit = self._ui.get("log_edit")
        if log_edit and msg:
            log_edit.appendPlainText(msg)
        lbl = self._ui.get("status_msg")
        if lbl:
            old = lbl.styleSheet()
            lbl.setStyleSheet(f"color:{color}; font-weight:bold;")
            lbl.setText(msg)
            QtCore.QTimer.singleShot(ms, lambda: lbl.setStyleSheet(old or ""))
        with contextlib.suppress(Exception):
            log_event({"event": "status_highlight", "message": msg})
    except Exception:
        self._status(msg)

def _reset_preview(self):
    self._last_image_pil = None
    self._last_image_qt = None
    self._last_video_frames_qt.clear()
    self._last_video_frames_pil.clear()
    if "preview_label" in self._ui: self._ui["preview_label"].setText("Preview")

def _display_qimage(self, qimage: QtGui.QImage):
    if not qimage or "preview_label" not in self._ui: return
    lbl = self._ui["preview_label"]
    pix = QtGui.QPixmap.fromImage(qimage)
    lbl.setPixmap(pix.scaled(
        lbl.size(),
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation
    ))
    # NEW: ensure/update star overlay
    try:
        self._ensure_star_button()
        self._update_star_button_state()
    except Exception:
        pass

def _pil_to_qimage(self, pil_img):
    from PIL import Image  # noqa
    if pil_img.mode not in ("RGB","RGBA"):
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", pil_img.mode)
    fmt = (QtGui.QImage.Format.Format_RGBA8888 if pil_img.mode == "RGBA"
           else QtGui.QImage.Format.Format_RGB888)
    return QtGui.QImage(data, pil_img.width, pil_img.height, fmt).copy()

# ---------- Refiner preload (idle-safe) ----------
def _maybe_preload_refiner(self):
    if getattr(self, "_busy", False):
        return
    if not hasattr(self, "_refiner_preload_queued"):
        self._refiner_preload_queued = False
    if self._refiner_preload_queued:
        return
    try:
        from src.modules import refiner_module as _ref
        _ref.ensure_loaded_async()
        self._refiner_preload_queued = True
    except Exception:
        pass

# ---------- Worker cleanup ----------
def _cleanup_worker(self, worker: "_Worker"):
    if worker in self._workers:
        self._workers.discard(worker)
    with contextlib.suppress(Exception):
        worker.deleteLater()

# ---------- Upscale ----------
def _populate_upscaler_combo(self):
    combo = self._ui.get("upscaler_combo")
    if not combo:
        return
    combo.blockSignals(True)
    combo.clear()
    combo.addItem("None")
    try:
        from src.modules.upscale_module import UPSCALE_MODEL_PATHS
        for name, path in UPSCALE_MODEL_PATHS.items():
            try:
                p = Path(path)
                if p.exists():
                    combo.addItem(name, str(p))
            except Exception:
                continue
    except Exception:
        try:
            from src.modules.upscale_module import UPSCALE_WEIGHTS_DIR
            root = Path(UPSCALE_WEIGHTS_DIR)
            for f in sorted(root.glob("*.pth")):
                combo.addItem(f.stem, str(f))
        except Exception:
            pass
    combo.blockSignals(False)

def _update_upscale_enable(self):
    combo = self._ui.get("upscaler_combo")
    btn = self._ui.get("btn_upscale")
    if not btn:
        return
    sel = combo.currentText() if combo else "None"
    has_image = self._last_image_pil is not None
    btn.setEnabled(sel != "None" and has_image and not self._busy)

def on_upscale(self):
    combo = self._ui.get("upscaler_combo")
    if not combo or combo.currentText() == "None":
        self._status("Select a Real-ESRGAN model to upscale.")
        return
    if self._last_image_pil is None:
        self._status("No image to upscale.")
        return
    model_path = combo.currentData()
    if not model_path or not Path(model_path).exists():
        self._status("Upscaler model file not found.")
        return
    self._status("Upscaling...")
    try:
        from src.modules.upscale_module import upscale_image_by_path
        up_img = upscale_image_by_path(self._last_image_pil, str(model_path), scale=4, device="cuda", save=True)
        if up_img is None:
            self._status("Upscale failed."); return
        self._last_image_pil = up_img
        self._last_image_qt = self._pil_to_qimage(up_img)
        self._display_qimage(self._last_image_qt)
        self._status("Upscaled (x4).")
        log_event({"event":"upscale_done"})
        with contextlib.suppress(Exception):
            self._add_to_gallery(up_img)
    except Exception as e:
        log_exception(e, context="upscale")
        self._status(f"Upscale error: {e}")


def on_show_vram(self):
    try:
        import importlib
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            self._status("CUDA not available.")
            return
        alloc = int(torch.cuda.memory_allocated() // 1024**2)
        reserved = int(torch.cuda.memory_reserved() // 1024**2)
        total = int(torch.cuda.get_device_properties(0).total_memory // 1024**2)
        self._status(f"VRAM: alloc {alloc} MB / reserved {reserved} MB / total {total} MB")
    except Exception as e:
        log_exception(e, context="show_vram")
        self._status(f"VRAM query failed: {e}")

def _start_video_preview(self):
    frames = getattr(self, "_last_video_frames_qt", [])
    if not frames:
        return
    self._video_frame_index = 0
    self._display_qimage(frames[0])
    try:
        fps = int(self._ui.get("video_fps_spin").value()) if "video_fps_spin" in self._ui else 8
        interval = max(1, int(1000 / max(1, fps)))
    except Exception:
        interval = 125
    try:
        if self._video_timer:
            self._video_timer.stop()
    except Exception:
        pass
    if not self._video_timer:
        self._video_timer = QtCore.QTimer(self)
        def _tick():
            try:
                if not self._last_video_frames_qt:
                    return
                self._video_frame_index = (self._video_frame_index + 1) % len(self._last_video_frames_qt)
                self._display_qimage(self._last_video_frames_qt[self._video_frame_index])
            except Exception:
                pass
        self._video_timer.timeout.connect(_tick)  # type: ignore
    self._video_timer.start(interval)

def _maybe_auto_save_images(self, pil_list: List[Any]):
    return

def _maybe_auto_save_video(self, pil_frames: List[Any]):
    return

def _maybe_auto_release_after_job(self):
    return

def _unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    for i in range(1, 10000):
        cand = base.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
    return base

def _default_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def on_save_image(self):
    if self._last_image_pil is None:
        self._status("No image to save.")
        return
    out_dir = Path(self._output_dir or "outputs/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use previous filename if any, else create a timestamped one
    name = (self._last_saved_path.name if self._last_saved_path
            else f"img_{_default_timestamp()}.png")
    target = _unique_path(out_dir / name)
    try:
        fmt = target.suffix.lower()
        params = {"quality": 95} if fmt in (".jpg", ".jpeg") else {}
        self._last_image_pil.save(str(target), **params)
        self._last_saved_path = target
        self._status(f"Saved: {target}")
        # Remember output dir if checkbox is on
        chk = self._ui.get("remember_save_chk")
        if chk and chk.isChecked():
            self._cfg["last_output_dir"] = str(out_dir)
            with contextlib.suppress(Exception):
                from src.modules.config_store import save_gui_config
                save_gui_config(self._cfg)
    except Exception as e:
        log_exception(e, context="save_image")
        self._status(f"Save failed: {e}")

def on_save_image_as(self):
    if self._last_image_pil is None:
        self._status("No image to save.")
        return
    out_dir = Path(self._output_dir or "outputs/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    dlg = QtW.QFileDialog(self, "Save Image As", str(out_dir),
                          "PNG (*.png);;JPEG (*.jpg *.jpeg);;WEBP (*.webp)")
    dlg.setAcceptMode(QtW.QFileDialog.AcceptMode.AcceptSave)
    if not dlg.exec():  # cancelled
        return
    path = Path(dlg.selectedFiles()[0])
    if not path.suffix:
        path = path.with_suffix(".png")
    try:
        fmt = path.suffix.lower()
        params = {"quality": 95} if fmt in (".jpg", ".jpeg") else {}
        path.parent.mkdir(parents=True, exist_ok=True)
        self._last_image_pil.save(str(path), **params)
        self._last_saved_path = path
        self._status(f"Saved: {path}")
        chk = self._ui.get("remember_save_chk")
        if chk and chk.isChecked():
            self._cfg["last_output_dir"] = str(path.parent)
            with contextlib.suppress(Exception):
                from src.modules.config_store import save_gui_config
                save_gui_config(self._cfg)
    except Exception as e:
        log_exception(e, context="save_image_as")
        self._status(f"Save failed: {e}")

def _write_gif(path: Path, frames_pil: list, fps: int):
    import numpy as np
    import imageio
    imgs = [np.array(f.convert("RGB")) for f in frames_pil]
    duration = 1.0 / max(1, int(fps or 8))
    imageio.mimsave(str(path), imgs, duration=duration)

def _write_mp4(path: Path, frames_pil: list, fps: int):
    import numpy as np
    import imageio
    writer = imageio.get_writer(str(path), fps=max(1, int(fps or 8)), codec="libx264", quality=8)
    try:
        for f in frames_pil:
            writer.append_data(np.array(f.convert("RGB")))
    finally:
        writer.close()

def on_save_video(self):
    if not self._last_video_frames_pil:
        self._status("No video to save.")
        return
    videos_dir = Path("outputs/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    # Prefer GIF by default to avoid ffmpeg requirement
    path = _unique_path(videos_dir / f"vid_{_default_timestamp()}.gif")
    fps = 8
    try:
        # Try to read FPS from UI if available
        if "video_fps_spin" in self._ui:
            fps = int(self._ui["video_fps_spin"].value())
    except Exception:
        pass
    try:
        try:
            _write_gif(path, self._last_video_frames_pil, fps)
            self._status(f"Saved: {path}")
        except Exception:
            alt = path.with_suffix(".mp4")
            _write_mp4(alt, self._last_video_frames_pil, fps)
            self._status(f"Saved: {alt}")
    except Exception as e:
        # Fallback: dump frames folder
        try:
            frames_dir = _unique_path(videos_dir / f"vid_{_default_timestamp()}_frames")
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(self._last_video_frames_pil):
                f.save(str(frames_dir / f"frame_{i:04d}.png"))
            self._status(f"Saved frames: {frames_dir}")
        except Exception as e2:
            log_exception(e2, context="save_video_frames_fallback")
            log_exception(e, context="save_video")
            self._status(f"Save failed: {e}")

def on_save_video_as(self):
    if not self._last_video_frames_pil:
        self._status("No video to save.")
        return
    videos_dir = Path("outputs/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    dlg = QtW.QFileDialog(self, "Save Video As", str(videos_dir),
                          "GIF (*.gif);;MP4 (*.mp4)")
    dlg.setAcceptMode(QtW.QFileDialog.AcceptMode.AcceptSave)
    if not dlg.exec():
        return
    path = Path(dlg.selectedFiles()[0])
    if path.suffix.lower() not in (".gif", ".mp4"):
        path = path.with_suffix(".gif")
    try:
        fps = 8
        if "video_fps_spin" in self._ui:
            fps = int(self._ui["video_fps_spin"].value())
        if path.suffix.lower() == ".gif":
            _write_gif(path, self._last_video_frames_pil, fps)
        else:
            _write_mp4(path, self._last_video_frames_pil, fps)
        self._status(f"Saved: {path}")
    except Exception as e:
        log_exception(e, context="save_video_as")
        self._status(f"Save failed: {e}")

# Compatibility wrapper referenced in _on_generate_done
def _update_upscaler_combo(self):
    try:
        self._update_upscale_enable()
    except Exception:
        pass


def _update_model_kind_enable(self):
    """
    Update UI enablement based on the selected model kind (image/video) and busy state.
    Ensures Refine is enabled only for image mode with an available image.
    """
    try:
        # Derive current model kind from combo, fallback to stored kind.
        kind = getattr(self, "_current_model_kind", "image")
        combo = self._ui.get("model_combo")
        if combo:
            data = combo.currentData()
            if isinstance(data, dict):
                kind = (data.get("kind") or kind) or "image"
        self._current_model_kind = kind

        # Apply button enablement via the existing helpers.
        if self._busy:
            self._set_buttons_running()
        else:
            self._set_buttons_idle()

        # Refine enablement only in image mode with a current image.
        can_refine = (kind == "image") and (self._last_image_pil is not None) and not self._busy
        act_ref = self._ui.get("act_refine")
        btn_ref = self._ui.get("btn_refine")
        if act_ref:
            act_ref.setEnabled(bool(can_refine))
        if btn_ref:
            btn_ref.setEnabled(bool(can_refine))
        # If refine is not applicable, stop attention and clear styling.
        if not can_refine and hasattr(self, "_stop_refine_attention"):
            self._stop_refine_attention(disable=True)
    except Exception:
        # Best-effort: avoid breaking UI on any error
        pass


def _update_generate_action_style(self):
    """
    Make Generate controls green when model is ready for this mode and UI is idle.
    """
    try:
        is_image = getattr(self, "_current_model_kind", "image") == "image"
        ready_ui = bool(is_image and not self._busy and getattr(self, "_model_ready", False))

        style_on = "background:#2e7d32; color:white; font-weight:bold;"
        style_off = ""

        tbtn = self._get_toolbar_button("act_generate")
        if tbtn:
            tbtn.setStyleSheet(style_on if ready_ui else style_off)

        btn = self._ui.get("btn_generate")
        if btn:
            btn.setStyleSheet(style_on if ready_ui else style_off)
    except Exception:
        pass


def _make_side_by_side_qimage(self, left: "QtGui.QImage", right: "QtGui.QImage") -> "QtGui.QImage":
    """
    Compose two QImages side-by-side with a small gap on a dark background.
    """
    try:
        if left is None or right is None:
            return left or right  # fallback
        gap = 8
        w = max(1, int(left.width())) + gap + max(1, int(right.width()))
        h = max(int(left.height()), int(right.height()))
        out = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
        out.fill(QtGui.QColor(32, 32, 32))  # dark neutral background
        painter = QtGui.QPainter(out)
        try:
            painter.setRenderHints(QtGui.QPainter.RenderHint.SmoothPixmapTransform | QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.drawImage(0, 0, left)
            painter.drawImage(int(left.width()) + gap, 0, right)
        finally:
            painter.end()
        return out
    except Exception:
        # In case of any error, return left or right to avoid crashes
        return left or right


def _show_before_after_popup(self, left: "QtGui.QImage", right: "QtGui.QImage"):
    """
    Show a non-modal dialog presenting the Before/After images.
    """
    try:
        if left is None or right is None:
            return
        if not hasattr(self, "_before_after_dialog") or self._before_after_dialog is None:
            self._before_after_dialog = _BeforeAfterDialog(self)
        dlg = self._before_after_dialog
        dlg.set_images(left, right)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
    except Exception:
        pass

# --- Inpaint editor wiring (stack + fullscreen) ---


def _ensure_central_stack(self):
    # If already installed, nothing to do
    if getattr(self, "_central_stack", None):
        return

    # If the current central is already a stack, reuse it
    cw = self.centralWidget()
    if isinstance(cw, QtW.QStackedWidget):
        self._central_stack = cw
        # Best-effort: assume index 0 is the original main widget
        with contextlib.suppress(Exception):
            self._central_main = cw.widget(0)
        return

    # Important: takeCentralWidget() detaches without deleting it
    central = self.takeCentralWidget()
    if central is None:
        return  # nothing to wrap (shouldn't happen if apply_ui ran)

    stack = QtW.QStackedWidget(self)
    stack.addWidget(central)
    self.setCentralWidget(stack)

    self._central_main = central
    self._central_stack = stack

def _install_inpaint_action(self):
    tb = self._ui.get("toolbar_main")
    if not tb:
        return
    if "act_inpaint" in self._ui:
        return
    act = QtGui.QAction("Inpaint", self)
    act.setToolTip("Open SDXL Inpaint Editor")
    act.setShortcut(QtGui.QKeySequence("Ctrl+I"))
    tb.addAction(act)
    self._ui["act_inpaint"] = act
    act.triggered.connect(self.on_open_inpaint_editor)  # type: ignore

def _hide_chrome_for_editor(self, on: bool):
    # Hide/show toolbars/docks to let the editor feel full-screen
    for key in ("toolbar_main",):
        w = self._ui.get(key)
        with contextlib.suppress(Exception):
            if w: w.setVisible(not on)
    # Docks (e.g., gallery)
    for key in ("gallery_dock",):
        d = self._ui.get(key)
        with contextlib.suppress(Exception):
            if d: d.setVisible(not on)

def on_open_inpaint_editor(self):
    try:
        from src.gui.inpaint_editor import InpaintEditorWidget
        self._ensure_central_stack()
        if not hasattr(self, "_inpaint_editor") or self._inpaint_editor is None:
            self._inpaint_editor = InpaintEditorWidget(self)
            # Back returns to main view
            self._inpaint_editor.backRequested.connect(self.on_close_inpaint_editor)  # type: ignore
            self._central_stack.addWidget(self._inpaint_editor)
        self._central_stack.setCurrentWidget(self._inpaint_editor)

        # Enter full-screen editing
        self._was_fullscreen = self.isFullScreen()
        self._was_maximized = self.isMaximized()
        self._hide_chrome_for_editor(True)
        self.showFullScreen()
        self._status("Inpaint editor opened.")
        log_event({"event": "inpaint_editor_opened"})
    except Exception as e:
        log_exception(e, context="open_inpaint_editor")
        self._status(f"Inpaint open failed: {e}")

def on_close_inpaint_editor(self):
    try:
        stack = getattr(self, "_central_stack", None)
        if stack:
            main = getattr(self, "_central_main", None)
            if main is not None and stack.indexOf(main) != -1:
                stack.setCurrentWidget(main)
            else:
                # Fallback to first page if our cached pointer was invalidated
                stack.setCurrentIndex(0)

        # Restore chrome and window state
        self._hide_chrome_for_editor(False)
        # Restore prior window state
        if not getattr(self, "_was_fullscreen", False):
            self.showNormal()
            if getattr(self, "_was_maximized", False):
                self.showMaximized()

        self._status("Returned to main window.")
        log_event({"event": "inpaint_editor_closed"})
    except Exception as e:
        log_exception(e, context="close_inpaint_editor")
        self._status(f"Close failed: {e}")

def _ensure_star_button(self):
    lbl = self._ui.get("preview_label")
    if not lbl:
        return
    btn = self._ui.get("star_btn")
    if not btn:
        btn = QtW.QToolButton(lbl)
        btn.setText("★")
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("QToolButton { background: rgba(0,0,0,0.0); border: none; font-size: 22px; color:#777; }")
        btn.setToolTip("Add to favourites")
        btn.clicked.connect(self._on_star_clicked)  # type: ignore
        btn.setVisible(False)
        self._ui["star_btn"] = btn
        with contextlib.suppress(Exception):
            lbl.installEventFilter(self)
    _reposition_star_btn(self)

def _reposition_star_btn(self):
    lbl = self._ui.get("preview_label")
    btn = self._ui.get("star_btn")
    if not lbl or not btn:
        return
    margin = 8
    btn.adjustSize()
    x = max(0, lbl.width() - btn.width() - margin)
    y = margin
    btn.move(x, y)

def _is_current_favourite(self) -> bool:
    qimg = getattr(self, "_last_image_qt", None)
    if qimg is None:
        return False
    for it in getattr(self, "_favorites_items", []):
        if it.get("qimg") is qimg:
            return True
    return False

def _update_star_button_state(self):
    btn = self._ui.get("star_btn")
    if not btn:
        return
    has_img = getattr(self, "_last_image_qt", None) is not None
    btn.setVisible(bool(has_img))
    if not has_img:
        return
    fav = _is_current_favourite(self)
    if fav:
        btn.setStyleSheet("QToolButton { background: rgba(0,0,0,0.0); border:none; font-size:22px; color:#FFD700; }")
        btn.setToolTip("Remove from favourites")
    else:
        btn.setStyleSheet("QToolButton { background: rgba(0,0,0,0.0); border:none; font-size:22px; color:#777; }")
        btn.setToolTip("Add to favourites")

def _on_star_clicked(self):
    try:
        if getattr(self, "_last_image_pil", None) is None or getattr(self, "_last_image_qt", None) is None:
            return
        self._init_gallery()
        if _is_current_favourite(self):
            _remove_favourite_by_qimg(self, self._last_image_qt)
            self._status("Removed from favourites.")
        else:
            _add_to_favourites(self, self._last_image_pil, self._last_image_qt)
            self._status("Added to favourites.")
            with contextlib.suppress(Exception):
                if "gallery_dock" in self._ui:
                    self._ui["gallery_dock"].show()
        _update_star_button_state(self)
    except Exception as e:
        log_exception(e, context="toggle_favourite")

def _add_to_favourites(self, pil_img, qimg: QtGui.QImage):
    try:
        self._init_gallery()
        fav = self._ui.get("fav_widget")
        if not fav:
            return
        icon = QtGui.QIcon(QtGui.QPixmap.fromImage(
            qimg.scaled(96,96,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation)))
        item = QtW.QListWidgetItem(icon, f"{len(self._favorites_items)+1}")
        fav.addItem(item)
        self._favorites_items.append({"pil": pil_img, "qimg": qimg, "item": item})
    except Exception as e:
        log_exception(e, context="favourites_add")

def _remove_favourite_by_qimg(self, qimg: QtGui.QImage):
    try:
        fav = self._ui.get("fav_widget")
        if not fav:
            return
        idx = next((i for i, it in enumerate(self._favorites_items) if it.get("qimg") is qimg), -1)
        if idx >= 0:
            it = self._favorites_items.pop(idx)
            row = fav.row(it["item"])
            if row >= 0:
                fav.takeItem(row)
    except Exception as e:
        log_exception(e, context="favourites_remove")

def _on_fav_item_clicked(self, item):
    try:
        fav = self._ui.get("fav_widget")
        if not fav:
            return
        idx = fav.row(item)
        if 0 <= idx < len(self._favorites_items):
            entry = self._favorites_items[idx]
            self._last_image_pil = entry["pil"]
            self._last_image_qt = entry["qimg"]
            self._display_qimage(self._last_image_qt)
            if "btn_refine" in self._ui:
                self._ui["btn_refine"].setEnabled(True)
    except Exception as e:
        log_exception(e, context="favourites_click")

def _on_fav_item_double_clicked(self, item):
    try:
        fav = self._ui.get("fav_widget")
        if not fav:
            return
        idx = fav.row(item)
        if 0 <= idx < len(self._favorites_items):
            _open_favourites_viewer(self, idx)
    except Exception as e:
        log_exception(e, context="favourites_double_click")

def _open_favourites_viewer(self, index: int):
    if not getattr(self, "_favorites_items", None):
        return
    if self._gallery_viewer is None:
        self._gallery_viewer = _GalleryViewer(self)
    self._gallery_viewer.set_items(self._favorites_items, index)
    self._gallery_viewer.show()
    self._gallery_viewer.raise_()
    self._gallery_viewer.activateWindow()

def _mw_event_filter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
    try:
        if watched is self._ui.get("preview_label") and event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Show,
            QtCore.QEvent.Type.Move,
        ):
            _reposition_star_btn(self)
    except Exception:
        pass
    return False
def on_save_video(self):
    if not self._last_video_frames_pil:
        self._status("No video to save.")
        return
    videos_dir = Path("outputs/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    # Prefer GIF by default to avoid ffmpeg requirement
    path = _unique_path(videos_dir / f"vid_{_default_timestamp()}.gif")
    fps = 8
    try:
        # Try to read FPS from UI if available
        if "video_fps_spin" in self._ui:
            fps = int(self._ui["video_fps_spin"].value())
    except Exception:
        pass
    try:
        try:
            _write_gif(path, self._last_video_frames_pil, fps)
            self._status(f"Saved: {path}")
        except Exception:
            alt = path.with_suffix(".mp4")
            _write_mp4(alt, self._last_video_frames_pil, fps)
            self._status(f"Saved: {alt}")
    except Exception as e:
        # Fallback: dump frames folder
        try:
            frames_dir = _unique_path(videos_dir / f"vid_{_default_timestamp()}_frames")
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(self._last_video_frames_pil):
                f.save(str(frames_dir / f"frame_{i:04d}.png"))
            self._status(f"Saved frames: {frames_dir}")
        except Exception as e2:
            log_exception(e2, context="save_video_frames_fallback")
            log_exception(e, context="save_video")
            self._status(f"Save failed: {e}")

# ----- Bind module-level helpers to MainWindow -----
try:
    _BINDINGS = {
        "_set_buttons_running": _set_buttons_running,
        "_set_buttons_idle": _set_buttons_idle,
        "_get_toolbar_button": _get_toolbar_button,
        "_update_cancel_button_style": _update_cancel_button_style,
        "_toggle_refine_flash": _toggle_refine_flash,
        "_start_refine_attention": _start_refine_attention,
        "_stop_refine_attention": _stop_refine_attention,
        "_status": _status,
        "_status_highlight": _status_highlight,
        "_reset_preview": _reset_preview,
        "_display_qimage": _display_qimage,
        "_pil_to_qimage": _pil_to_qimage,
        "_maybe_preload_refiner": _maybe_preload_refiner,
        "_cleanup_worker": _cleanup_worker,
        "_populate_upscaler_combo": _populate_upscaler_combo,
        "_update_upscale_enable": _update_upscale_enable,
        "on_upscale": on_upscale,
        "on_show_vram": on_show_vram,
        "_start_video_preview": _start_video_preview,
        "_maybe_auto_save_images": _maybe_auto_save_images,
        "_maybe_auto_save_video": _maybe_auto_save_video,
        "_maybe_auto_release_after_job": _maybe_auto_release_after_job,
        "on_save_image": on_save_image,
        "on_save_image_as": on_save_image_as,
        "on_save_video": on_save_video,
        "on_save_video_as": on_save_video_as,
        "_update_upscaler_combo": _update_upscaler_combo,
        "_update_model_kind_enable": _update_model_kind_enable,
        "_update_generate_action_style": _update_generate_action_style,
        "_make_side_by_side_qimage": _make_side_by_side_qimage,
        "_show_before_after_popup": _show_before_after_popup,
        "_ensure_central_stack": _ensure_central_stack,
        "_install_inpaint_action": _install_inpaint_action,
        "_hide_chrome_for_editor": _hide_chrome_for_editor,
        "on_open_inpaint_editor": on_open_inpaint_editor,
        "on_close_inpaint_editor": on_close_inpaint_editor,
        "_ensure_star_button": _ensure_star_button,
        "_reposition_star_btn": _reposition_star_btn,
        "_is_current_favourite": _is_current_favourite,
        "_update_star_button_state": _update_star_button_state,
        "_on_star_clicked": _on_star_clicked,
        "_add_to_favourites": _add_to_favourites,
        "_remove_favourite_by_qimg": _remove_favourite_by_qimg,
        "_on_fav_item_clicked": _on_fav_item_clicked,
        "_on_fav_item_double_clicked": _on_fav_item_double_clicked,
        "_open_favourites_viewer": _open_favourites_viewer,
        # Ensure Qt calls into our reposition logic
        "eventFilter": _mw_event_filter,
    }
    for _name, _fn in _BINDINGS.items():
        if not hasattr(MainWindow, _name):
            setattr(MainWindow, _name, _fn)
except Exception as _e:
    with contextlib.suppress(Exception):
        log_exception(_e, context="bind_helpers")
