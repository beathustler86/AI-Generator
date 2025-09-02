from __future__ import annotations
from typing import Optional, Any, List
import os, sys, threading
from pathlib import Path
import traceback

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
        # Diagnostics: verify expected handlers exist
        missing_handlers = [n for n in ("on_save_image","on_save_image_as") if not hasattr(self, n)]
        if missing_handlers:
            print(f"[Diagnostics] Missing handlers: {missing_handlers}")

        self._dark_mode = False
        self._busy = False
        self._cancel_event: Optional[threading.Event] = None
        self._workers: set[_Worker] = set()

        self._last_batch_pil: List[Any] = []
        self._last_image_pil = None
        self._last_image_qt = None
        self._last_video_frames_qt: List[QtGui.QImage] = []
        self._last_video_frames_pil: List[Any] = []
        self._video_timer: Optional[QtCore.QTimer] = None
        self._video_frame_index = 0
        self._current_model_kind = "image"

        self._output_dir = Path("outputs/images"); self._output_dir.mkdir(parents=True, exist_ok=True)
        self._last_saved_path = None
        # ---- Load GUI config ----
        self._cfg = load_gui_config()
        remember_flag = bool(self._cfg.get("remember_output_dir", True))
        if "remember_save_chk" in self._ui:
            self._ui["remember_save_chk"].setChecked(remember_flag)
        cache_flag = bool(self._cfg.get("use_pipeline_disk_cache", True))
        if "pipeline_cache_chk" in self._ui:
            self._ui["pipeline_cache_chk"].setChecked(cache_flag)
        if hasattr(gen_mod, "set_disk_cache_enabled") and hasattr(gen_mod, "is_disk_cache_enabled"):
            try:
                if gen_mod.is_disk_cache_enabled() != cache_flag:
                    gen_mod.set_disk_cache_enabled(cache_flag)
            except Exception as e:
                log_exception(e, context="disk_cache_toggle_init")
        else:
            # Diagnostic: generation module probably partially imported
            print("[Diagnostics] generation.set_disk_cache_enabled missing (partial import?).", flush=True)
            log_event({"event":"generation_partial_import","missing":"set_disk_cache_enabled"})
        if remember_flag:
            cfg_dir = self._cfg.get("last_output_dir")
            if cfg_dir:
                try:
                    self._output_dir = Path(cfg_dir)
                    self._output_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
        last_model = self._cfg.get("last_model_path")
        if last_model:
            os.environ["MODEL_ID_OR_PATH"] = last_model
        # Restore window geometry/state if present
        try:
            geo = self._cfg.get("window_geometry")
            state = self._cfg.get("window_state")
            if isinstance(geo, list):
                self.restoreGeometry(bytes(geo))
            if isinstance(state, list):
                self.restoreState(bytes(state))
        except Exception:
            pass

        self._gallery_items: List[dict] = []
        # Gallery directory setup
        self._gallery_dir = Path(
            self._cfg.get("gallery_dir")
            or os.environ.get("GALLERY_DIR")
            or r"C:\Users\beath\Documents\MyRepos\GitRepos\gallery"
        )
        self._gallery_dir.mkdir(parents=True, exist_ok=True)
        auto_g = bool(self._cfg.get("auto_save_to_gallery", True))
        if "auto_gallery_chk" in self._ui:
            self._ui["auto_gallery_chk"].setChecked(auto_g)

        self._populate_model_combo()
        self._set_buttons_idle()   # now safe
        log_event({"event": "ui_init"})
        # Connect prompt edits for dynamic style
        self._ui["prompt_edit"].textChanged.connect(self._update_generate_action_style)  # type: ignore
        self._ui["negative_edit"].textChanged.connect(self._update_generate_action_style)  # type: ignore
        QtCore.QTimer.singleShot(0, self._update_generate_action_style)
        # Ensure gallery docks exist early if desired (lazy by default)
        # self._init_gallery()  # uncomment if you prefer gallery always visible on startup
        # If model population failed earlier due to missing method, it's now present below.
        # Load existing gallery thumbnails (non-blocking) shortly after UI shows
        QtCore.QTimer.singleShot(250, self._initial_gallery_scan)
        # Start VRAM telemetry timer (every 20s)
        self._vram_timer = QtCore.QTimer(self)
        self._vram_timer.timeout.connect(self._vram_telemetry_tick)  # type: ignore
        self._vram_timer.start(20000)

        self._last_vram_tuple = None  # add near other state fields

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
            # sort newest first, cap load
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            files = files[:50]
            if not files:
                return
            self._init_gallery()
            from PIL import Image
            for f in reversed(files):  # older first so numbering is chronological
                try:
                    im = Image.open(f).convert("RGB")
                    self._add_to_gallery(im)
                except Exception:
                    pass
            log_event({"event":"gallery_scan","count":len(files)})
        except Exception as e:
            log_exception(e, context="gallery_scan")

    # ---------- VRAM telemetry ----------
    def _vram_telemetry_tick(self):
        try:
            import importlib
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                return
            torch.cuda.synchronize()
            alloc = int(torch.cuda.memory_allocated() // 1024**2)
            reserved = int(torch.cuda.memory_reserved() // 1024**2)
            total = int(torch.cuda.get_device_properties(0).total_memory // 1024**2)
            has_pipe = False
            try:
                if hasattr(gen_mod, "has_pipeline"):
                    has_pipe = gen_mod.has_pipeline()
                else:
                    has_pipe = getattr(gen_mod, "_PIPELINE", None) is not None
            except Exception:
                has_pipe = False
            tup = (alloc,reserved,has_pipe,self._current_model_kind)
            if tup != getattr(self, "_last_vram_tuple", None) or os.environ.get("VRAM_SNAPSHOT_DEDUP","1") == "0":
                from src.modules.utils.telemetry import log_event
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

    # ---------- Helper UI state methods (added) ----------
    def _set_buttons_running(self):
        if "act_generate" in self._ui: self._ui["act_generate"].setEnabled(False)
        if "btn_generate" in self._ui: self._ui["btn_generate"].setEnabled(False)
        if "act_generate_video" in self._ui: self._ui["act_generate_video"].setEnabled(False)
        if "act_cancel" in self._ui: self._ui["act_cancel"].setEnabled(True)
        if "btn_cancel" in self._ui: self._ui["btn_cancel"].setEnabled(True)

    def _set_buttons_idle(self):
        if "act_generate" in self._ui:
            self._ui["act_generate"].setEnabled(self._current_model_kind == "image" and not self._busy)
        if "btn_generate" in self._ui:
            self._ui["btn_generate"].setEnabled(self._current_model_kind == "image" and not self._busy)
        if "act_generate_video" in self._ui:
            self._ui["act_generate_video"].setEnabled(self._current_model_kind == "video" and not self._busy)
        if "act_cancel" in self._ui: self._ui["act_cancel"].setEnabled(False)
        if "btn_cancel" in self._ui: self._ui["btn_cancel"].setEnabled(False)
        # Refresh styling after state change
        if hasattr(self, "_update_generate_action_style"):
            self._update_generate_action_style()

    def _update_generate_action_style(self):
        """
        Color the toolbar Generate actions green only when BOTH prompt and negative boxes have text.
        Does not disable actions (only visual cue). Adjust here if you want to enforce enabling logic.
        """
        prompt_ok = bool(self._ui.get("prompt_edit") and self._ui["prompt_edit"].toPlainText().strip())
        negative_ok = bool(self._ui.get("negative_edit") and self._ui["negative_edit"].toPlainText().strip())
        active = prompt_ok and negative_ok
        style_on = "QToolButton { background:#2e7d32; color:white; font-weight:bold; }"
        style_off = ""
        tb = self._ui.get("toolbar_main")
        if tb:
            for key in ("act_generate","act_generate_video"):
                act = self._ui.get(key)
                if not act: continue
                btn = tb.widgetForAction(act)
                if btn:
                    btn.setStyleSheet(style_on if active else style_off)

    def _update_model_kind_enable(self):
        """
        Enable / disable actions & widgets based on current model kind (image/video),
        busy state, and availability of last results.
        """
        is_video = (self._current_model_kind == "video")
        if "video_row_container" in self._ui:
            self._ui["video_row_container"].setVisible(is_video)

        if "act_generate" in self._ui:
            self._ui["act_generate"].setEnabled(not is_video and not self._busy)
        if "act_generate_video" in self._ui:
            self._ui["act_generate_video"].setEnabled(is_video and not self._busy)
        if "btn_generate" in self._ui:
            self._ui["btn_generate"].setEnabled(not is_video and not self._busy)
        if "act_refine" in self._ui:
            self._ui["act_refine"].setEnabled(not is_video and self._last_image_pil is not None)
        if "act_upscale" in self._ui:
            self._ui["act_upscale"].setEnabled(
                not is_video and self._last_image_pil is not None and not self._busy
            )
        if "btn_cancel" in self._ui:
            self._ui["btn_cancel"].setEnabled(self._busy)
        # Image save actions
        if "act_save_image" in self._ui:
            img_ok = (not is_video and self._last_image_pil is not None)
            self._ui["act_save_image"].setEnabled(img_ok)
            if "act_save_image_as" in self._ui:
                self._ui["act_save_image_as"].setEnabled(img_ok)
        # Video save actions
        if "act_save_video" in self._ui:
            vid_ok = (is_video and bool(self._last_video_frames_pil))
            self._ui["act_save_video"].setEnabled(vid_ok)
            if "act_save_video_as" in self._ui:
                self._ui["act_save_video_as"].setEnabled(vid_ok)
        # Optional export actions
        if "act_export_gif" in self._ui:
            has_frames = bool(self._last_video_frames_pil)
            self._ui["act_export_gif"].setEnabled(is_video and has_frames)
            if "act_export_mp4" in self._ui:
                self._ui["act_export_mp4"].setEnabled(is_video and has_frames)

        # Preview label placeholders
        if is_video and not self._last_video_frames_qt and "preview_label" in self._ui:
            self._ui["preview_label"].setText("Video Preview")
        elif (not is_video) and not self._last_image_qt and "preview_label" in self._ui:
            self._ui["preview_label"].setText("Preview")

        # Update generate button styling after state adjustments
        self._update_generate_action_style()

    def _status(self, msg: str):
        if "status_msg" in self._ui:
            self._ui["status_msg"].setText(msg)
        log_edit = self._ui.get("log_edit")
        if log_edit and msg and not msg.startswith(("Sampling step","Frame ")):
            log_edit.appendPlainText(msg)
        try:
            if msg:
                log_event({"event":"status","message":msg})
        except Exception:
            pass

    def _reset_preview(self):
        self._last_image_pil = None
        self._last_image_qt = None
        self._last_video_frames_qt.clear()
        self._last_video_frames_pil.clear()
        if "preview_label" in self._ui:
            self._ui["preview_label"].setText("Preview")

    def _display_qimage(self, qimage: QtGui.QImage):
        if not qimage or "preview_label" not in self._ui:
            return
        lbl = self._ui["preview_label"]
        pix = QtGui.QPixmap.fromImage(qimage)
        lbl.setPixmap(
            pix.scaled(
                lbl.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
        )

    def _pil_to_qimage(self, pil_img):
        from PIL import Image  # noqa
        if pil_img.mode not in ("RGB","RGBA"):
            pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw", pil_img.mode)
        fmt = (QtGui.QImage.Format.Format_RGBA8888
               if pil_img.mode == "RGBA"
               else QtGui.QImage.Format.Format_RGB888)
        return QtGui.QImage(data, pil_img.width, pil_img.height, fmt).copy()

    # ---------- Gallery helpers (added) ----------
    def _init_gallery(self):
        if "gallery_widget" in self._ui:
            return
        dock = QtW.QDockWidget("Gallery", self)
        tabs = QtW.QTabWidget()
        lst = QtW.QListWidget()
        lst.setIconSize(QtCore.QSize(96,96))
        lst.itemClicked.connect(self._on_gallery_item_clicked)  # type: ignore
        tabs.addTab(lst, "Gallery")
        dock.setWidget(tabs)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._ui["gallery_dock"] = dock
        self._ui["gallery_tabs"] = tabs
        self._ui["gallery_widget"] = lst

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

    def _on_gallery_item_clicked(self, item):
        try:
            idx = self._ui["gallery_widget"].row(item)
            if 0 <= idx < len(self._gallery_items):
                self._last_image_pil = self._gallery_items[idx]["pil"]
                self._last_image_qt = self._gallery_items[idx]["qimg"]
                self._display_qimage(self._last_image_qt)
                if "btn_refine" in self._ui:
                    self._ui["btn_refine"].setEnabled(True)
        except Exception as e:
            log_exception(e, context="gallery_click")
    # Fallback stub if somehow removed
    if not hasattr(__qualname__, "_on_gallery_item_clicked"):
        def _on_gallery_item_clicked(self, item):  # pragma: no cover
            pass

    # ---------- Model handling ----------
    def _populate_model_combo(self):
        combo = self._ui.get("model_combo")
        if not combo:
            return
        combo.clear()
        try:
            entries = gen_mod.list_all_models()
        except Exception:
            entries = []
        # Optional force refresh if env flag set
        if os.getenv("REFRESH_MODEL_LIST","0")=="1":
            from importlib import reload
            reload(gen_mod)
            try:
                entries = gen_mod.list_all_models()
            except Exception:
                pass
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
            })
        try:
            current_target = gen_mod.current_model_target()
        except Exception:
            current_target = None
        if current_target:
            for i in range(combo.count()):
                data = combo.itemData(i)
                if data and data.get("path") == current_target:
                    combo.setCurrentIndex(i)
                    break
        self._update_model_kind_enable()

    def on_load_model(self):
        combo = self._ui.get("model_combo")
        if not combo:
            self._status("Model combo missing.")
            return
        data = combo.currentData()
        if not data:
            self._status("No model selected.")
            return
        kind = data.get("kind","image")
        path = data.get("path")
        if not path:
            self._status("Model path missing.")
            return
        self._current_model_kind = kind
        self._status("Scheduling model load...")
        self._busy = True
        self._set_buttons_running()
        self.repaint()
        # Worker thread for load
        sampler = self._ui.get("sampler_combo").currentText() if "sampler_combo" in self._ui else "euler_a"
        parent = self
        class _ModelLoader(_Worker):
            def __init__(self):
                super().__init__(fn=self._do, args=(), parent=parent)
            def _do(self, progress_cb):
                try:
                    progress_cb("Setting target...")
                    gen_mod.set_model_target(path)
                    os.environ["MODEL_ID_OR_PATH"] = path
                    progress_cb("Loading pipeline (async)...")
                    gen_mod.force_load_pipeline(sampler=sampler, device="cuda")
                    try: gen_mod.trigger_background_prefetch()
                    except Exception: pass
                    return True
                except Exception as ex:
                    log_exception(ex, context="async_model_load_internal")
                    raise
        worker = _ModelLoader()
        def _progress(msg): self._status(msg)
        worker.progress.connect(_progress)  # type: ignore
        def _done(_):
            self._reset_preview()
            log_event({"event":"model_loaded","model":path,"kind":kind})
            self._status(f"Model loaded: {data.get('label', path)}")
            self._busy = False
            self._set_buttons_idle()
            self._update_model_kind_enable()
            self._update_generate_action_style()
            self._vram_telemetry_tick()
        def _err(e: Exception):
            log_exception(e, context="async_model_load")
            print("[ModelLoader][ERROR]", "".join(traceback.format_exception(type(e), e, e.__traceback__)))
            self._status(f"Load failed: {e}")
            self._busy = False
            self._set_buttons_idle()
        worker.result.connect(_done)  # type: ignore
        worker.error.connect(_err)    # type: ignore
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))  # type: ignore
        log_event({"event":"model_loader_spawn","path":path,"sampler":sampler})
        worker.started.connect(lambda: log_event({"event":"model_loader_thread_started"}))  # type: ignore
        self._workers.add(worker)
        worker.start()

    # ---------- Image Generation ----------
    def on_generate(self):
        if self._busy or self._current_model_kind != "image":
            return
        # Auto randomize seed if checkbox enabled
        auto_chk = self._ui.get("seed_auto_chk")
        if auto_chk and auto_chk.isChecked():
            try:
                import random
                new_seed = random.randint(1, 2**31 - 2)
                self._ui["seed_spin"].setValue(new_seed)
                log_event({"event":"seed_autorand","seed":new_seed})
            except Exception:
                pass
        self._busy = True
        self._cancel_event = threading.Event()
        self._set_buttons_running()
        ui = self._ui
        args = dict(
            prompt=ui["prompt_edit"].toPlainText().strip(),
            negative=ui["negative_edit"].toPlainText().strip(),
            steps=ui["steps_spin"].value(),
            cfg=ui["cfg_spin"].value(),
            w=ui["width_spin"].value(),
            h=ui["height_spin"].value(),
            seed=ui["seed_spin"].value(),
            batch=ui["batch_spin"].value(),
            sampler=ui["sampler_combo"].currentText()
        )
        self._status("Preparing generation...")
        worker = _Worker(fn=self._do_generate_image, args=(args,), parent=self)
        worker.progress.connect(self._on_generate_progress)
        worker.result.connect(self._on_generate_done)
        worker.error.connect(self._on_generate_error)
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))
        self._workers.add(worker)
        worker.start()

    # ---------- Video Generation ----------
    def on_generate_video(self):
        if self._busy or self._current_model_kind != "video":
            return
        self._busy = True
        self._cancel_event = threading.Event()
        self._set_buttons_running()
        prompt = self._ui["prompt_edit"].toPlainText().strip()
        negative = self._ui["negative_edit"].toPlainText().strip()
        frames = self._ui["video_frames_spin"].value()
        fps = self._ui["video_fps_spin"].value()
        width = self._ui["width_spin"].value()
        height = self._ui["height_spin"].value()
        steps = self._ui["steps_spin"].value()
        cfg = self._ui["cfg_spin"].value()
        seed = self._ui["seed_spin"].value()
        self._status("Preparing video...")
        log_event({"event":"video_generation_start","frames":frames,"fps":fps,"w":width,"h":height,"steps":steps,"cfg":cfg})
        worker = _Worker(
            fn=self._do_generate_video,
            args=(prompt, negative, frames, width, height, fps, steps, cfg, seed, self._cancel_event),
            parent=self
        )
        worker.progress.connect(self._on_generate_progress)
        worker.result.connect(self._on_video_done)
        worker.error.connect(self._on_generate_error)
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))
        self._workers.add(worker)
        worker.start()

    # ---------- Refine ----------
    def on_refine(self):
        if self._current_model_kind == "video":
            self._status("Refine not supported for video.")
            return
        if self._last_image_pil is None:
            self._status("No image to refine.")
            return
        try:
            self._status("Refining...")
            log_event({"event":"refine_start"})
            from importlib import import_module
            ref_mod = import_module("src.modules.refiner_module")
            # Prefer latent path later; for now fallback RGB:
            fn = getattr(ref_mod, "refine_image_from_rgb", None)
            if not fn:
                self._status("Refiner unavailable.")
                return
            refined = fn(self._last_image_pil,
                         prompt=self._ui["prompt_edit"].toPlainText().strip(),
                         negative_prompt=self._ui["negative_edit"].toPlainText().strip() or None)
            if refined is None:
                self._status("Refine failed.")
                log_event({"event":"refine_fail"})
                return
            self._last_image_pil = refined
            self._last_image_qt = self._pil_to_qimage(refined)
            self._display_qimage(self._last_image_qt)
            self._status("Refined.")
            log_event({"event":"refine_success"})
        except Exception as e:
            self._status(f"Refine failed: {e}")
            log_exception(e, context="refine")

    # ---------- Video preset handlers ----------
    def on_video_preset_changed(self, text: str):
        if text == "Custom": return
        try:
            w, h = map(int, text.lower().split("x"))
        except Exception:
            return
        self._ui["width_spin"].blockSignals(True)
        self._ui["height_spin"].blockSignals(True)
        self._ui["width_spin"].setValue(w)
        self._ui["height_spin"].setValue(h)
        self._ui["width_spin"].blockSignals(False)
        self._ui["height_spin"].blockSignals(False)
        self._status(f"Video resolution preset applied: {w}x{h}")

    def on_video_dims_manual_changed(self, *_):
        combo = self._ui.get("video_preset_combo")
        if not combo: return
        w = self._ui["width_spin"].value()
        h = self._ui["height_spin"].value()
        expected = f"{w}x{h}"
        idx = combo.findText(expected)
        combo.blockSignals(True)
        if idx >= 0: combo.setCurrentIndex(idx)
        else:
            custom = combo.findText("Custom")
            if custom >= 0: combo.setCurrentIndex(custom)
        combo.blockSignals(False)

    def _annotate_backend_state(self, backend_state: str):
        if backend_state == "stub":
            self._status("Video backend: stub (bouncing ball) – add Cosmos code & weights.")
        elif backend_state == "real_pending":
            self._status("Video backend: real_pending (noise placeholder) – set COSMOS_WEIGHTS & real model.")
        elif backend_state == "real_ready":
            self._status("Video backend: real_ready (model active).")

    # ---------- Worker callable implementations ----------
    def _do_generate_image(self, args: dict, progress_cb):
        prompt=args["prompt"]; negative=args["negative"]; steps=args["steps"]; cfg=args["cfg"]
        w=args["w"]; h=args["h"]; seed=args["seed"]; batch=args["batch"]; sampler=args["sampler"]
        cancel = self._cancel_event
        progress_cb("Loading pipeline...")
        if os.environ.get("SHOW_CACHE_STATUS","1") == "1":
            try:
                from src.modules.generation import disk_cache_root
                cache_root = disk_cache_root()
                cur = gen_mod.current_model_target()
                if cache_root.replace("\\","/") in cur.replace("\\","/"):
                    self._status("Using disk-cached pipeline.")
            except Exception:
                pass
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
        # Convert each PIL image safely
        qimg_list: List[Any] = []
        for im in imgs_pil:
            try:
                qimg_list.append(gen_mod.pil_to_qimage(im))
            except Exception as conv_err:
                log_exception(conv_err, context="qimage_convert")
        log_event({"event":"generation","count":len(qimg_list),"steps":steps,"cfg":cfg,"w":w,"h":h,"sampler":sampler})
        progress_cb("Finalizing")
        return (imgs_pil, qimg_list)

    def _do_generate_video(self, prompt: str, negative: str, frames: int, w: int, h: int,
                           fps: int, steps: int, cfg: float, seed: int,
                           cancel_event, progress_cb):
        frames_pil = gen_mod.generate_video(
            prompt=prompt, negative=negative, frames=frames,
            width=w, height=h, fps=fps, steps=steps, cfg=cfg, seed=seed,
            progress_cb=progress_cb, cancel_event=cancel_event
        )
        qframes = [gen_mod.pil_to_qimage(f) for f in frames_pil]
        progress_cb("Finalizing")
        log_event({"event":"video_generation","frames":len(qframes),"steps":steps,"cfg":cfg,"w":w,"h":h})
        return (frames_pil, qframes)

    # ---------- Worker callbacks ----------
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
                try:
                    for im in pil_list:
                        self._add_to_gallery(im)
                    # Auto show gallery after first successful add
                    if "gallery_dock" in self._ui:
                        self._ui["gallery_dock"].show()
                except Exception as e:
                    log_exception(e, context="gallery_add")
                if "btn_refine" in self._ui:
                    self._ui["btn_refine"].setEnabled(True)
                self._status(f"Generated {len(qimg_list)} image(s).")
                self._maybe_auto_save_images(pil_list)
        finally:
            self._busy = False
            self._set_buttons_idle()
            self._update_model_kind_enable()
            self._maybe_auto_release_after_job()
            self._vram_telemetry_tick()

    def _on_video_done(self, result):
        pil_frames, q_frames = result if isinstance(result, tuple) else ([], result)
        if not q_frames:
            self._status("Video generation failed.")
        else:
            self._last_video_frames_pil = pil_frames
            self._last_video_frames_qt = q_frames
            self._start_video_preview()
            self._status(f"Generated video ({len(q_frames)} frames).")
            log_event({"event":"video_generation_complete","frames":len(q_frames)})
            self._maybe_auto_save_video(pil_frames)
        self._busy = False
        self._set_buttons_idle()
        self._update_model_kind_enable()
        self._maybe_auto_release_after_job()
        self._vram_telemetry_tick()

    def _on_generate_progress(self, msg: str):
        if os.environ.get("VERBOSE_TELEMETRY") == "1":
            log_event({"event":"progress","message":msg})
        self._status(msg)

    def _on_generate_error(self, err: Exception):
        if str(err).lower() == "cancelled":
            if self._current_model_kind == "video":
                log_event({"event":"video_cancelled"})
            else:
                log_event({"event":"image_cancelled"})
            self._status("Cancelled.")
        else:
            log_exception(err, context="generate_worker")
            self._status(f"Error: {err}")
        self._busy = False
        self._set_buttons_idle()
        self._update_model_kind_enable()

    # ---------- Video preview ----------
    def _start_video_preview(self):
        self._stop_video_preview()
        if not self._last_video_frames_qt:
            return
        self._video_timer = QtCore.QTimer(self)
        self._video_timer.timeout.connect(self._advance_video_frame)  # type: ignore
        interval = self._current_video_interval_ms()
        self._video_timer.start(interval)
        log_event({"event":"video_preview_start","interval_ms":interval})
        self._advance_video_frame()

    def _advance_video_frame(self):
        if not self._last_video_frames_qt: return
        self._video_frame_index = (self._video_frame_index + 1) % len(self._last_video_frames_qt)
        self._display_qimage(self._last_video_frames_qt[self._video_frame_index])
        if os.environ.get("VERBOSE_TELEMETRY") == "1":
            log_event({"event":"video_preview_frame","index":self._video_frame_index})
        if self._video_timer:
            desired = self._current_video_interval_ms()
            if self._video_timer.interval() != desired:
                self._video_timer.setInterval(desired)

    def _stop_video_preview(self):
        if self._video_timer:
            self._video_timer.stop()
            self._video_timer.deleteLater()
            self._video_timer = None

    # ---------- Cancel ----------
    def on_cancel_generate(self):
        if self._busy and self._cancel_event and not self._cancel_event.is_set():
            self._cancel_event.set()
            self._status("Cancelling...")

    # ---------- File / Dir ----------
    def on_open_models_dir(self):
        path = getattr(getattr(self.services,"model_manager",None), "models_root", None)
        if not path:
            self._status("No models root configured.")
            return
        import subprocess, sys
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            self._status("Opened models directory.")
        except Exception as e:
            self._status(f"Open failed: {e}")

    def on_open_settings(self):
        self._status("Settings dialog not implemented yet.")

    def on_clear_log(self):
        self._ui["log_edit"].clear()
        self._status("Log cleared.")

    def on_show_vram(self):
        try:
            import importlib, subprocess
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                alloc = torch.cuda.memory_allocated() // 1024**2
                reserved = torch.cuda.memory_reserved() // 1024**2
                total = torch.cuda.get_device_properties(0).total_memory // 1024**2
                if total == 0 or (alloc == 0 and reserved == 0):
                    try:
                        result = subprocess.run(
                            ["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],
                            capture_output=True,text=True,timeout=2
                        )
                        if result.returncode==0:
                            used_csv = result.stdout.strip().split(os.linesep)
                            for line in used_csv:
                                alloc, total = map(int, (x.strip() for x in line.split(",")))
                                if alloc > 0 and total > 0:
                                    break
                    except Exception:
                        pass
                self._ui["status_vram"].setText(f"VRAM: {alloc} / {reserved} / {total} MB (alloc/resv/total)")
            else:
                self._ui["status_vram"].setText("CPU mode")
        except Exception as e:
            self._status(f"VRAM query failed: {e}")
            log_exception(e, context="vram")

    # ---------- Cache / VRAM management ----------
    def on_release_vram(self):
        try:
            from src.modules import generation as gen_mod
            gen_mod.release_pipeline(free_ram=False)
            self._status("Pipeline moved to CPU; VRAM freed.")
        except Exception as e:
            self._status(f"Release failed: {e}")

    def on_release_vram_full(self):
        try:
            from src.modules import generation as gen_mod
            gen_mod.release_pipeline(free_ram=True)
            self._status("Pipeline fully released.")
        except Exception as e:
            self._status(f"Full release failed: {e}")

    def on_cache_info(self):
        try:
            from src.modules import generation as gen_mod
            stats = gen_mod.get_cache_stats()
            lines = [
                f"Root: {stats['root']}",
                f"Enabled: {stats['enabled']}",
                f"Entries: {stats['count']}  Total: {stats['total_mb']} MB"
            ]
            for ent in stats["entries"]:
                lines.append(f"- {ent['slug']} {ent.get('mb','?')}MB {ent.get('files','?')} files")
            QtW.QMessageBox.information(self, "Pipeline Cache", "\n".join(lines))
            self._status("Cache info shown.")
        except Exception as e:
            self._status(f"Cache info error: {e}")

    def on_cache_purge(self):
        ret = QtW.QMessageBox.question(self,"Confirm","Purge all cached pipelines?")
        if ret != QtW.QMessageBox.StandardButton.Yes:
            return
        try:
            from src.modules import generation as gen_mod
            gen_mod.purge_disk_cache()
            self._status("Cache purged.")
        except Exception as e:
            self._status(f"Purge failed: {e}")

    def on_toggle_auto_release(self):
        self._cfg["auto_release_enabled"] = not bool(self._cfg.get("auto_release_enabled"))
        try: save_gui_config(self._cfg)
        except Exception: pass
        self._status(f"Auto release {'enabled' if self._cfg['auto_release_enabled'] else 'disabled'}.")

    def _start_auto_release_timer(self):
        self._stop_auto_release_timer()
        minutes = max(1, int(self._cfg.get("auto_release_minutes", 10)))
        interval_ms = minutes * 60 * 1000
        self._auto_release_timer = QtCore.QTimer(self)
        self._auto_release_timer.timeout.connect(self._on_auto_release_tick)  # type: ignore
        self._auto_release_timer.start(interval_ms)

    def _stop_auto_release_timer(self):
        if self._auto_release_timer:
            self._auto_release_timer.stop()
            self._auto_release_timer.deleteLater()
            self._auto_release_timer = None

    def _on_auto_release_tick(self):
        if self._busy:
            return
        mode = self._cfg.get("auto_release_mode","cpu")
        try:
            gen_mod.release_pipeline(free_ram=(mode=="free"))
            log_event({"event":"pipeline_auto_release","mode":mode})
            self._status(f"Auto release: {mode}.")
        except Exception as e:
            self._status(f"Auto release failed: {e}")

    # ---------- Post-job VRAM release helper ----------
    def _maybe_auto_release_after_job(self):
        if not self._cfg.get("auto_release_enabled"):
            return
        mode = self._cfg.get("auto_release_mode","cpu")
        try:
            gen_mod.release_pipeline(free_ram=(mode=="free"))
            log_event({"event":"pipeline_post_job_release","mode":mode})
            self._status(f"VRAM released ({mode}).")
        except Exception as e:
            log_exception(e, context="post_job_release")

    # ---------- Worker cleanup ----------
    def _cleanup_worker(self, worker: "_Worker"):
        if worker in self._workers:
            self._workers.discard(worker)
        try:
            worker.deleteLater()
        except Exception:
            pass

    # ---------- Upscaler & Remember Output Helpers ----------
    def on_toggle_remember_output(self, checked: bool):
        self._cfg["remember_output_dir"] = bool(checked)
        save_gui_config(self._cfg)

    def on_toggle_pipeline_cache(self, checked: bool):
        gen_mod.set_disk_cache_enabled(bool(checked))
        self._cfg["use_pipeline_disk_cache"] = bool(checked)
        try:
            save_gui_config(self._cfg)
        except Exception:
            pass
        self._status(f"Pipeline cache {'enabled' if checked else 'disabled'}.")
        log_event({"event":"pipeline_cache_toggle","enabled":bool(checked)})
    def on_toggle_auto_gallery(self, checked: bool):
        self._cfg["auto_save_to_gallery"] = bool(checked)
        save_gui_config(self._cfg)
        self._status(f"Auto gallery {'ON' if checked else 'OFF'}.")
        log_event({"event":"auto_gallery_toggle","enabled":bool(checked)})
    def on_toggle_gallery(self):
        self._init_gallery()
        dock = self._ui.get("gallery_dock")
        if dock:
            vis = not dock.isVisible()
            dock.setVisible(vis)
            self._status(f"Gallery {'shown' if vis else 'hidden'}.")
    def on_upscale_last(self):
        if self._current_model_kind == "video":
            self._status("Upscale only for images.")
            return
        if self._last_image_pil is None:
            self._status("No image to upscale.")
            return
        self._status("Upscaling...")
        try:
            up_img = self._run_realesrgan(self._last_image_pil)
            if up_img is None:
                self._status("Upscale failed.")
                return
            self._last_image_pil = up_img
            self._last_image_qt = self._pil_to_qimage(up_img)
            self._display_qimage(self._last_image_qt)
            self._status("Upscaled (x4).")
            log_event({"event":"upscale_done"})
            try:
                self._add_to_gallery(up_img)
            except Exception:
                pass
        except Exception as e:
            log_exception(e, context="upscale")
            self._status(f"Upscale error: {e}")

    def _run_realesrgan(self, pil_img):
        import sys, torch
        from torchvision import transforms
        from src.modules import generation as gen_mod
        entries = gen_mod.list_upscalers()
        if not entries:
            raise RuntimeError("No upscaler models found (models/Upscaler/*.pth)")
        entries.sort(key=lambda e: (0 if "ultra" in e["name"].lower() else 1,
                                     0 if ("x4" in e["name"].lower() or "4x" in e["name"].lower()) else 2,
                                     e["name"]))
        model_path = entries[0]["path"]
        up_root = Path(model_path).parent.parent
        if str(up_root) not in sys.path:
            sys.path.insert(0, str(up_root))
        try:
            from realesrgan.archs.rrdbnet import RRDBNet
        except Exception:
            raise RuntimeError("RealESRGAN package not installed (pip install realesrgan basicsr).")
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                       num_grow_ch=32, scale=4)
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "params_ema" in state:
            state = state["params_ema"]
        missing = net.load_state_dict(state, strict=False)
        if getattr(missing, "missing_keys", None):
            print("[Upscale] Missing keys:", missing.missing_keys)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net.to(device).eval()
        tfm = transforms.ToTensor()
        inp = tfm(pil_img).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = net(inp)
        out = out.clamp(0,1)[0].cpu()
        arr = (out.permute(1,2,0).numpy()*255).round().astype("uint8")
        from PIL import Image
        return Image.fromarray(arr)

    # ---------- Auto-save helpers ----------
    def _auto_gallery_enabled(self) -> bool:
        return bool(self._cfg.get("auto_save_to_gallery", True))

    def _default_filename_base(self):
        from datetime import datetime
        seed = self._ui["seed_spin"].value() if "seed_spin" in self._ui else 0
        return f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_seed{seed}"

    def _ensure_output_dir(self):
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def on_set_output_dir(self):
        dlg = QtW.QFileDialog(self, "Select Output Directory")
        dlg.setFileMode(QtW.QFileDialog.FileMode.Directory)
        dlg.setOption(QtW.QFileDialog.Option.ShowDirsOnly, True)
        if dlg.exec():
            sel = dlg.selectedFiles()
            if sel:
                self._output_dir = Path(sel[0])
                self._ensure_output_dir()
                if self._ui.get("remember_save_chk", None) and self._ui["remember_save_chk"].isChecked():
                    self._cfg["last_output_dir"] = str(self._output_dir)
                    try: save_gui_config(self._cfg)
                    except Exception: pass
                self._status(f"Output dir set: {self._output_dir}")

    def on_save_image(self):
        if self._last_image_pil is None:
            self._status("No image to save.")
            return
        self._ensure_output_dir()
        base = self._default_filename_base()
        path = self._output_dir / f"{base}.png"
        try:
            self._last_image_pil.save(path)
            self._last_saved_path = path
            log_event({"event":"save_image","path":str(path)})
            self._status(f"Saved: {path.name}")
            if self._ui.get("remember_save_chk", None) and self._ui["remember_save_chk"].isChecked():
                self._cfg["last_output_dir"] = str(self._output_dir)
                try: save_gui_config(self._cfg)
                except Exception: pass
        except Exception as e:
            log_exception(e, context="save_image")
            self._status(f"Save failed: {e}")

    def on_save_image_as(self):
        if self._last_image_pil is None:
            self._status("No image to save.")
            return
        self._ensure_output_dir()
        base = self._default_filename_base()
        fn, _ = QtW.QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            str(self._output_dir / f"{base}.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WEBP (*.webp);;All Files (*.*)"
        )
        if not fn:
            return
        try:
            ext = Path(fn).suffix.lower()
            save_params = {}
            if ext in (".jpg",".jpeg"):
                save_params["quality"] = 95
            self._last_image_pil.save(fn, **save_params)
            self._last_saved_path = Path(fn)
            log_event({"event":"save_image_as","path":fn})
            self._status(f"Saved: {Path(fn).name}")
            if self._ui.get("remember_save_chk", None) and self._ui["remember_save_chk"].isChecked():
                self._cfg["last_output_dir"] = str(Path(fn).parent)
                try: save_gui_config(self._cfg)
                except Exception: pass
        except Exception as e:
            log_exception(e, context="save_image_as")
            self._status(f"Save failed: {e}")

    def _collect_video_frames(self):
        return self._last_video_frames_pil if self._last_video_frames_pil else []

    def on_save_video(self):
        frames = self._collect_video_frames()
        if not frames:
            self._status("No video frames to save.")
            return
        self._ensure_output_dir()
        from datetime import datetime
        base = self._default_filename_base()
        path = self._output_dir / f"{base}.gif"
        try:
            fps = max(1, self._ui.get("video_fps_spin").value()) if "video_fps_spin" in self._ui else 8
            dur = int(1000 / fps)
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0)
            log_event({"event":"save_video","path":str(path),"frames":len(frames),"fps":fps})
            self._status(f"Video saved: {path.name}")
        except Exception as e:
            log_exception(e, context="save_video")
            self._status(f"Save video failed: {e}")

    def on_save_video_as(self):
        frames = self._collect_video_frames()
        if not frames:
            self._status("No video frames to save.")
            return
        self._ensure_output_dir()
        base = self._default_filename_base()
        fn, _ = QtW.QFileDialog.getSaveFileName(
            self,
            "Save Video As",
            str(self._output_dir / f"{base}.gif"),
            "GIF (*.gif);;All Files (*.*)"
        )
        if not fn:
            return
        try:
            fps = max(1, self._ui.get("video_fps_spin").value()) if "video_fps_spin" in self._ui else 8
            dur = int(1000 / fps)
            frames[0].save(fn, save_all=True, append_images=frames[1:], duration=dur, loop=0)
            log_event({"event":"save_video_as","path":fn,"frames":len(frames),"fps":fps})
            self._status(f"Video saved: {Path(fn).name}")
        except Exception as e:
            log_exception(e, context="save_video_as")
            self._status(f"Save video failed: {e}")

    def on_randomize_seed(self):
        import random
        sb = self._ui.get("seed_spin")
        if not sb:
            self._status("Seed widget missing.")
            return
        new_seed = random.randint(1, 2**31 - 2)
        sb.setValue(new_seed)
        log_event({"event":"seed_randomized","seed":new_seed})
        self._status(f"Seed randomized: {new_seed}")

    # ---------- NEW: Auto-save implementations ----------
    def _maybe_auto_save_images(self, pil_list: List[Any]):
        if not pil_list or not self._auto_gallery_enabled():
            return
        try: self._gallery_dir.mkdir(parents=True, exist_ok=True)
        except Exception: pass
        base_ts = self._default_filename_base()
        prompt_txt = self._ui["prompt_edit"].toPlainText().strip() if "prompt_edit" in self._ui else ""
        snippet = ("_" + "".join(c for c in prompt_txt[:32] if c.isalnum() or c in "-_").lower()) if prompt_txt else ""
        imgs = list(pil_list)
        def _worker():
            import threading
            log_event({"event":"gallery_async_start","count":len(imgs)})
            saved_paths=[]
            for idx, im in enumerate(imgs, start=1):
                fname=f"{base_ts}_{idx}{snippet}.png"
                path=self._gallery_dir / fname
                try:
                    im.save(path)
                    saved_paths.append(str(path))
                except Exception as e:
                    log_exception(e, context="gallery_async_save")
            log_event({"event":"gallery_async_done","saved":len(saved_paths)})
        threading.Thread(target=_worker, daemon=True).start()
        self._status(f"Auto-saving {len(imgs)} image(s) in background...")

    def _maybe_auto_save_video(self, frames: List[Any]):
        if not frames or not self._auto_gallery_enabled():
            return
        try: self._gallery_dir.mkdir(parents=True, exist_ok=True)
        except Exception: pass
        base_ts=self._default_filename_base()
        fps = max(1, self._ui.get("video_fps_spin").value()) if "video_fps_spin" in self._ui else 8
        dur=int(1000/fps)
        def _worker():
            fname=f"{base_ts}_anim.gif"
            path=self._gallery_dir / fname
            try:
                frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0)
                log_event({"event":"gallery_async_video_done","frames":len(frames),"path":str(path),"fps":fps})
            except Exception as e:
                log_event({"event":"gallery_async_video_fail"})
                log_exception(e, context="auto_gallery_video_save")
        import threading
        threading.Thread(target=_worker, daemon=True).start()
        self._status("Auto-saving video (background)...")
# --- Restored Worker Thread Base -------------------------------------------------
try:
    from PySide6 import QtCore as _QtCore  # type: ignore
except Exception:
    try:
        from PyQt5 import QtCore as _QtCore  # type: ignore
    except Exception:
        _QtCore = None

if _QtCore and "_Worker" not in globals():
    # Select correct Signal factory (PySide6 uses Signal, PyQt5 uses pyqtSignal)
    SignalType = getattr(_QtCore, "Signal", getattr(_QtCore, "pyqtSignal", None))

    class _Worker(_QtCore.QThread):
        """
        Generic background worker thread.
        Emits:
          progress(str)
          result(object)
          error(Exception)
        """
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
            try:
                self.progress.emit(msg)
            except Exception:
                pass
