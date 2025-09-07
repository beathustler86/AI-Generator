from __future__ import annotations

from typing import Optional
import contextlib

try:
    from PySide6 import QtWidgets as QtW, QtCore, QtGui
except ImportError:  # pragma: no cover
    from PyQt5 import QtWidgets as QtW, QtCore, QtGui  # type: ignore

from PIL import Image
from src.modules.inpainting_modules.sdxl_inpainting_module import (
    InpaintingCanvas,
    InpaintConfig,
    inpaint_image,
    safe_open_image,
)
from src.modules.utils.telemetry import log_event, log_exception


class _RunWorker(QtCore.QThread):
    progress = QtCore.Signal(str)      # type: ignore
    result = QtCore.Signal(object)     # type: ignore
    error = QtCore.Signal(Exception)   # type: ignore

    def __init__(self, canvas: InpaintingCanvas, cfg: InpaintConfig, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._cfg = cfg

    def run(self):
        try:
            imgs = inpaint_image(self._canvas, self._cfg, progress_cb=self._emit)
            self.result.emit(imgs)
        except Exception as e:
            self.error.emit(e)

    def _emit(self, msg: str):
        with contextlib.suppress(Exception):
            self.progress.emit(msg)


class InpaintEditorWidget(QtW.QWidget):
    backRequested = QtCore.Signal()  # type: ignore

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._canvas: Optional[InpaintingCanvas] = None
        self._cfg = InpaintConfig()

        # Toolbar row
        bar = QtW.QHBoxLayout()
        self._btn_back = QtW.QPushButton("⟵ Back")
        self._btn_open = QtW.QPushButton("Open Image")
        self._btn_overlay = QtW.QPushButton("Toggle Overlay")
        self._btn_run = QtW.QPushButton("Run Inpaint")
        bar.addWidget(self._btn_back)
        bar.addStretch(1)
        bar.addWidget(self._btn_open)
        bar.addWidget(self._btn_overlay)
        bar.addWidget(self._btn_run)

        # Simple prompt row (scaffold)
        row2 = QtW.QHBoxLayout()
        self._edit_prompt = QtW.QLineEdit()
        self._edit_prompt.setPlaceholderText("Prompt")
        self._edit_negative = QtW.QLineEdit()
        self._edit_negative.setPlaceholderText("Negative prompt")
        self._steps = QtW.QSpinBox(); self._steps.setRange(1, 200); self._steps.setValue(self._cfg.steps)
        self._cfg_spin = QtW.QDoubleSpinBox(); self._cfg_spin.setRange(0.0, 25.0); self._cfg_spin.setValue(self._cfg.cfg); self._cfg_spin.setSingleStep(0.1)
        self._denoise = QtW.QDoubleSpinBox(); self._denoise.setRange(0.0, 1.0); self._denoise.setValue(self._cfg.denoise_strength); self._denoise.setSingleStep(0.05)
        row2.addWidget(QtW.QLabel("Prompt:")); row2.addWidget(self._edit_prompt, 2)
        row2.addWidget(QtW.QLabel("Neg:")); row2.addWidget(self._edit_negative, 2)
        row2.addWidget(QtW.QLabel("Steps:")); row2.addWidget(self._steps)
        row2.addWidget(QtW.QLabel("CFG:")); row2.addWidget(self._cfg_spin)
        row2.addWidget(QtW.QLabel("Denoise:")); row2.addWidget(self._denoise)

        # Preview
        self._preview = QtW.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet("QLabel { background:#202020; color:#DDD; }")

        lay = QtW.QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addLayout(row2)
        lay.addWidget(self._preview, 1)

        # Wire buttons
        self._btn_back.clicked.connect(self.backRequested.emit)   # type: ignore
        self._btn_open.clicked.connect(self._on_open)             # type: ignore
        self._btn_overlay.clicked.connect(self._on_toggle_overlay)  # type: ignore
        self._btn_run.clicked.connect(self._on_run)               # type: ignore

    # ---- Drag & Drop ----
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self._load_image(path)
                break

    # ---- Actions ----
    def _on_open(self):
        path, _ = QtW.QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self._load_image(path)

    def _on_toggle_overlay(self):
        if not self._canvas: return
        self._canvas.toggle_overlay(not self._canvas.overlay.show)
        self._update_preview()

    def _on_run(self):
        if not self._canvas:
            QtW.QMessageBox.information(self, "Inpaint", "Load an image first.")
            return
        # Update config from UI
        self._cfg.prompt = self._edit_prompt.text().strip()
        self._cfg.negative_prompt = self._edit_negative.text().strip()
        self._cfg.steps = int(self._steps.value())
        self._cfg.cfg = float(self._cfg_spin.value())
        self._cfg.denoise_strength = float(self._denoise.value())

        self._btn_run.setEnabled(False)
        self._btn_run.setText("Running...")
        self._worker = _RunWorker(self._canvas, self._cfg, parent=self)
        self._worker.progress.connect(lambda m: self._set_status(m))  # type: ignore
        self._worker.result.connect(self._on_done)                    # type: ignore
        self._worker.error.connect(self._on_err)                      # type: ignore
        self._worker.finished.connect(lambda: setattr(self, "_worker", None))  # type: ignore
        self._worker.start()

    def _on_done(self, imgs):
        try:
            if imgs:
                # Replace canvas base with first result to allow iterative edits
                out: Image.Image = imgs[0]
                self._canvas.image = out.convert("RGB")
                self._update_preview()
                self._set_status("Done.")
                with contextlib.suppress(Exception):
                    log_event({"event": "inpaint_editor_done"})
        finally:
            self._btn_run.setEnabled(True)
            self._btn_run.setText("Run Inpaint")

    def _on_err(self, e: Exception):
        log_exception(e, context="inpaint_editor_run")
        QtW.QMessageBox.critical(self, "Error", str(e))
        self._btn_run.setEnabled(True)
        self._btn_run.setText("Run Inpaint")

    # ---- Helpers ----
    def _load_image(self, path: str):
        try:
            im = safe_open_image(path, strip=True)
            self._canvas = InpaintingCanvas(im)
            self._update_preview()
            self._set_status(f"Loaded: {path}")
            log_event({"event": "inpaint_editor_open", "path": path})
        except Exception as e:
            log_exception(e, context="inpaint_editor_open")
            QtW.QMessageBox.critical(self, "Open Image", f"Failed: {e}")

    def _update_preview(self):
        if not self._canvas:
            self._preview.setText("Drop an image or click Open Image")
            return
        qimg = self._pil_to_qimage(self._canvas.get_overlay_rgba().convert("RGB"))
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self._preview.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self._preview.setPixmap(pix)

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        self._update_preview()

    @staticmethod
    def _pil_to_qimage(pil_img: Image.Image) -> QtGui.QImage:
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw", pil_img.mode)
        fmt = (QtGui.QImage.Format.Format_RGBA8888 if pil_img.mode == "RGBA" else QtGui.QImage.Format.Format_RGB888)
        return QtGui.QImage(data, pil_img.width, pil_img.height, fmt).copy()

    def _set_status(self, msg: str):
        if msg:
            # Simple inline status; could integrate a status bar later
            self._btn_run.setToolTip(msg)
