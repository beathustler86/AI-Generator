from __future__ import annotations
from typing import Optional, Any, List
import os, sys, threading
from pathlib import Path

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
        gen_mod.set_disk_cache_enabled(cache_flag)
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

        self._populate_model_combo()
        self._set_buttons_idle()   # now safe
        log_event({"event": "ui_init"})

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
        lst = QtW.QListWidget()
        lst.setIconSize(QtCore.QSize(96,96))
        lst.itemClicked.connect(self._on_gallery_item_clicked)  # type: ignore
        dock.setWidget(lst)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._ui["gallery_dock"] = dock
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
            log_exception(e, context="gallery_add_internal")

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

    # ---------- Model handling ----------
    def _populate_model_combo(self):
        combo = self._ui["model_combo"]
        combo.clear()
        for entry in gen_mod.list_all_models():
            combo.addItem(entry["label"], entry)
        combo.addItem("HF: SDXL Base (hub)", {
            "label":"HF: SDXL Base (hub)",
            "path":"stabilityai/stable-diffusion-xl-base-1.0",
            "kind":"image"
        })
        current = gen_mod.current_model_target()
        if current:
            for i in range(combo.count()):
                data = combo.itemData(i)
                if data and data.get("path") == current:
                    combo.setCurrentIndex(i)
                    break
        self._update_model_kind_enable()

    def on_load_model(self):
        combo = self._ui["model_combo"]
        data = combo.currentData()
        if not data:
            self._status("No model selected.")
            return
        kind = data.get("kind", "image")
        path = data.get("path")
        self._current_model_kind = kind
        if kind == "image":
            gen_mod.set_model_target(path)
            # Kick off background prefetch (optional)
            try:
                gen_mod.trigger_background_prefetch()
            except Exception:
                pass
        self._reset_preview()
        log_event({"event": "model_loaded", "model": path, "kind": kind})
        self._status(f"Model loaded: {data.get('label')}")
        self._update_model_kind_enable()

    def _update_model_kind_enable(self):
        is_video = (self._current_model_kind == "video")
        if "video_row_container" in self._ui:
            self._ui["video_row_container"].setVisible(is_video)

        self._ui["act_generate"].setEnabled(not is_video and not self._busy)
        self._ui["btn_generate"].setEnabled(not is_video and not self._busy)
        self._ui["act_generate_video"].setEnabled(is_video and not self._busy)
        self._ui["act_refine"].setEnabled(not is_video and self._last_image_pil is not None)
        if "act_upscale" in self._ui:
            self._ui["act_upscale"].setEnabled(not is_video and self._last_image_pil is not None and not self._busy)
        if "btn_cancel" in self._ui:
            self._ui["btn_cancel"].setEnabled(self._busy)
        # Image save actions
        if "act_save_image" in self._ui:
            img_ok = (not is_video and self._last_image_pil is not None)
            self._ui["act_save_image"].setEnabled(img_ok)
            self._ui["act_save_image_as"].setEnabled(img_ok)
        # Video save actions
        if "act_save_video" in self._ui:
            vid_ok = (is_video and bool(self._last_video_frames_pil))
            self._ui["act_save_video"].setEnabled(vid_ok)
            self._ui["act_save_video_as"].setEnabled(vid_ok)
        # Export (if present)
        if "act_export_gif" in self._ui:
            has_frames = bool(self._last_video_frames_pil)
            self._ui["act_export_gif"].setEnabled(is_video and has_frames)
            self._ui["act_export_mp4"].setEnabled(is_video and has_frames)

        if is_video and not self._last_video_frames_qt:
            self._ui["preview_label"].setText("Video Preview")
        elif not is_video and not self._last_image_qt:
            self._ui["preview_label"].setText("Preview")

    # ---------- Image Generation ----------
    def on_generate(self):
        if self._busy or self._current_model_kind != "image":
            return
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
            refine_image = getattr(ref_mod, "refine_image", None)
            if not refine_image:
                self._status("Refiner unavailable.")
                return
            result = refine_image(self._last_image_pil)
            refined = result.get("image")
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
        # If disk cache used, generation module logged an event; optionally surface
        if os.environ.get("SHOW_CACHE_STATUS","1") == "1":
            try:
                # crude check: if pipeline path is inside cache root
                from src.modules.generation import disk_cache_root, current_model_target
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
        qimgs = [gen_mod.pil_to_qimage(im) for im in imgs_pil]
        log_event({"event":"generation","count":len(qimgs),"steps":steps,"cfg":cfg,"w":w,"h":h,"sampler":sampler})
        progress_cb("Finalizing")
        return (imgs_pil, qimgs)

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
        return (frames_pil, qframes)

    # ---------- Worker callbacks ----------
    def _on_generate_done(self, result):
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
            except Exception as e:
                log_exception(e, context="gallery_add")
            self._ui["btn_refine"].setEnabled(True)
            self._status(f"Generated {len(qimg_list)} image(s).")
        self._busy = False
        self._set_buttons_idle()
        self._update_model_kind_enable()

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
            try:
                from src.nodes.cosmos_backend import backend_status
                self._annotate_backend_state(backend_status().get("backend","?"))
            except Exception:
                pass
        self._busy = False
        self._set_buttons_idle()
        self._update_model_kind_enable()

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
                            used_csv = result.stdout.strip().splitlines()[0].split(",")
                            alloc = int(used_csv[0].strip())
                            total = int(used_csv[1].strip())
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

    # ---------- Worker cleanup ----------
    def _cleanup_worker(self, worker: "_Worker"):
        if worker in self._workers:
            self._workers.discard(worker)
        try:
            worker.deleteLater()
        except Exception:
            pass

    # ---------- (ADD BELOW IF NOT PRESENT) Upscaler & Remember Output Helpers ----------
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
            from src.modules.utils.telemetry import log_event
            log_event({"event":"upscale_done"})
            try:
                self._add_to_gallery(up_img)
            except Exception:
                pass
        except Exception as e:
            from src.modules.utils.telemetry import log_exception
            log_exception(e, context="upscale")
            self._status(f"Upscale error: {e}")

    def _run_realesrgan(self, pil_img):
        """
        Minimal RealESRGAN RRDBNet x4 loading from models/Upscaler.
        Requires realesrgan/basicsr installed. Falls back with RuntimeError if missing.
        """
        import sys, torch
        from torchvision import transforms
        from src.modules import generation as gen_mod
        entries = gen_mod.list_upscalers()
        if not entries:
            raise RuntimeError("No upscaler models found (models/Upscaler/*.pth)")
        # priority: ultra / x4
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

class _Worker(QtCore.QThread):
    progress = QtCore.Signal(str)
    result  = QtCore.Signal(object)
    error   = QtCore.Signal(object)
    def __init__(self, fn, args, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._args = args
    def run(self):
        try:
            self.result.emit(self._fn(*self._args, progress_cb=lambda m: self.progress.emit(m)))
        except Exception as e:
            from src.modules.utils.telemetry import log_exception
            log_exception(e, context="worker_run")
            self.error.emit(e)

def _bootstrap():  # pragma: no cover
    import sys
    app = QtW.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":  # pragma: no cover
    _bootstrap()
