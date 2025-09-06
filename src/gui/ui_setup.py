from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Callable, Any
try:
    from PySide6 import QtWidgets as QtW, QtGui, QtCore
except ImportError:
    from PyQt5 import QtWidgets as QtW, QtGui, QtCore  # type: ignore
import torch; torch.cuda.empty_cache()

@dataclass
class Services:
    logger: Optional[Any] = None
    settings: Optional[Any] = None
    model_manager: Optional[Any] = None

def apply_ui(win: QtW.QMainWindow, services: Services) -> None:
    win._ui: Dict[str, Any] = {}
    _build_actions(win, services)
    _build_menus(win, services)
    _build_toolbars(win, services)
    _build_central(win, services)
    _build_docks(win, services)
    _build_statusbar(win, services)
    _apply_styles(win, qss_path="src/gui/resources/app.qss")

def _build_actions(win, services: Services) -> None:
    def mk(text, slot, shortcut=None, tip=None, icon=None):
        return _make_action(win, text=text, slot=slot, shortcut=shortcut, tip=tip, icon=icon)

    spec = [
        ("&Open Models Dir", "on_open_models_dir", "act_open_models", "Ctrl+M", "Open base models folder"),
        ("&Generate Image", "on_generate", "act_generate", "Ctrl+G", "Generate image(s)"),
        ("Generate &Video", "on_generate_video", "act_generate_video", None, "Generate video"),
        ("&Randomize Seed", "on_randomize_seed", "act_random_seed", "Ctrl+Shift+R", "Randomize seed value"),
        ("&Refine", "on_refine", "act_refine", "Ctrl+R", "Refine last image"),
        ("Se&ttings", "on_open_settings", "act_settings", "Ctrl+,", "Open application settings"),
        ("&Quit", "close", "act_quit", "Ctrl+Q", "Exit application"),
        ("Clear &Log", "on_clear_log", "act_clear_log", None, "Clear log panel"),
        ("Show &VRAM", "on_show_vram", "act_show_vram", "Ctrl+Shift+V", "Display VRAM usage"),
        ("Toggle &Theme", "on_toggle_theme", "act_toggle_dark", "Ctrl+T", "Toggle light/dark theme"),
        ("&Save Image", "on_save_image", "act_save_image", "Ctrl+S", "Save current image / frame"),
        ("Save Image &As...", "on_save_image_as", "act_save_image_as", "Ctrl+Shift+S", "Save image/frame as"),
        ("Save &Video", "on_save_video", "act_save_video", None, "Quick save video"),
        ("Save Video A&s...", "on_save_video_as", "act_save_video_as", None, "Save video with format selection"),
        ("Set &Output Dir...", "on_set_output_dir", "act_set_output_dir", None, "Choose output folder"),
        ("&Cancel", "on_cancel_generate", "act_cancel", "Esc", "Cancel active generation"),
        ("&Upscale Last Image (x4)", "on_upscale_last", "act_upscale", "Ctrl+U", "Upscale last image via RealESRGAN"),
        ("Release &VRAM (GPU)", "on_release_vram", "act_release_vram", None, "Move pipeline to CPU & free GPU memory"),
        ("Full Release (&RAM)", "on_release_vram_full", "act_release_vram_full", None, "Drop pipeline from RAM + GPU"),
        ("Cache &Info", "on_cache_info", "act_cache_info", None, "Show pipeline disk cache statistics"),
        ("&Purge Cache", "on_cache_purge", "act_cache_purge", None, "Delete all cached pipelines"),
        ("Auto &Release VRAM", "on_toggle_auto_release", "act_toggle_auto_release", None, "Toggle idle auto VRAM release"),
        ("Open &Cache Dir", "on_open_cache_dir", "act_open_cache_dir", None, "Open pipeline disk cache folder"),
        ("Show &Gallery", "on_toggle_gallery", "act_toggle_gallery", None, "Show/Hide gallery dock"),
    ]
    for label, attr, key, shortcut, tip in spec:
        if hasattr(win, attr):
            act = mk(label, getattr(win, attr), shortcut, tip)
            win._ui[key] = act
            if key == "act_upscale":
                act.setEnabled(False)

def _build_menus(win, services: Services) -> None:
    mb = win.menuBar()
    file_menu = mb.addMenu("&File")
    for key in ("act_open_models","act_settings"):
        if key in win._ui:
            file_menu.addAction(win._ui[key])
    file_menu.addSeparator()
    for key in ("act_save_image","act_save_image_as","act_save_video","act_save_video_as","act_set_output_dir"):
        if key in win._ui:
            file_menu.addAction(win._ui[key])
    file_menu.addSeparator()
    if "act_quit" in win._ui:
        file_menu.addAction(win._ui["act_quit"])

    gen_menu = mb.addMenu("&Generation")
    for key in ("act_generate","act_generate_video","act_refine"):
        if key in win._ui:
            gen_menu.addAction(win._ui[key])
    gen_menu.addSeparator()
    if "act_cancel" in win._ui:
        gen_menu.addAction(win._ui["act_cancel"])
    gen_menu.addSeparator()
    if "act_show_vram" in win._ui: gen_menu.addAction(win._ui["act_show_vram"])
    if "act_release_vram" in win._ui: gen_menu.addAction(win._ui["act_release_vram"])
    if "act_release_vram_full" in win._ui: gen_menu.addAction(win._ui["act_release_vram_full"])
    if any(k in win._ui for k in ("act_cache_info","act_cache_purge","act_toggle_auto_release","act_open_cache_dir")):
        gen_menu.addSeparator()
    if "act_cache_info" in win._ui: gen_menu.addAction(win._ui["act_cache_info"])
    if "act_cache_purge" in win._ui: gen_menu.addAction(win._ui["act_cache_purge"])
    if "act_open_cache_dir" in win._ui: gen_menu.addAction(win._ui["act_open_cache_dir"])
    if "act_toggle_auto_release" in win._ui: gen_menu.addAction(win._ui["act_toggle_auto_release"])

    view_menu = mb.addMenu("&View")
    if "act_toggle_dark" in win._ui: view_menu.addAction(win._ui["act_toggle_dark"])
    if "act_clear_log" in win._ui: view_menu.addAction(win._ui["act_clear_log"])
    if "act_toggle_gallery" in win._ui: view_menu.addAction(win._ui["act_toggle_gallery"])
    help_menu = mb.addMenu("&Help")

    win._ui.update(menu_file=file_menu, menu_generation=gen_menu,
                   menu_view=view_menu, menu_help=help_menu)

def _build_toolbars(win, services: Services) -> None:
    tb = win.addToolBar("Main")
    tb.setObjectName("MainToolbar")
    for key in ("act_generate","act_generate_video","act_cancel"):
        if key in win._ui:
            tb.addAction(win._ui[key])
    tb.addSeparator()
    if "act_refine" in win._ui:
        tb.addAction(win._ui["act_refine"])
    tb.addSeparator()
    if "act_toggle_dark" in win._ui: tb.addAction(win._ui["act_toggle_dark"])
    if "act_show_vram" in win._ui: tb.addAction(win._ui["act_show_vram"])
    if "act_save_image" in win._ui: tb.addAction(win._ui["act_save_image"])
    model_combo = QtW.QComboBox()
    model_combo.setMinimumWidth(260)
    win._ui["model_combo"] = model_combo
    load_btn = QtW.QToolButton()
    load_btn.setText("Load Model")
    if hasattr(win, "on_load_model"):
        load_btn.clicked.connect(win.on_load_model)  # type: ignore
    else:
        load_btn.setEnabled(False)
        load_btn.setToolTip("Model load handler unavailable")
    tb.addSeparator()
    tb.addWidget(model_combo)
    tb.addWidget(load_btn)
    win._ui["toolbar_main"] = tb

def _build_central(win, services: Services) -> None:
    central = QtW.QWidget()
    layout = QtW.QVBoxLayout(central)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)

    form = QtW.QFormLayout()
    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

    win._ui["prompt_edit"] = QtW.QPlainTextEdit()
    win._ui["negative_edit"] = QtW.QPlainTextEdit()
    win._ui["steps_spin"] = _spin_box(30, 1, 200)
    win._ui["cfg_spin"] = _double_spin_box(7.5, 1.0, 30.0, 0.1)
    win._ui["width_spin"] = _spin_box(1280, 64, 4096, 8)
    win._ui["height_spin"] = _spin_box(720, 64, 4096, 8)
    win._ui["seed_spin"] = _spin_box(0, 0, 2 ** 31 - 1)
    win._ui["batch_spin"] = _spin_box(1, 1, 64)
    win._ui["sampler_combo"] = QtW.QComboBox()
    win._ui["sampler_combo"].addItems(["euler_a", "ddim", "dpm++", "dpm-sde"])

    win._ui["video_frames_spin"] = _spin_box(16, 4, 1024, 4)
    win._ui["video_fps_spin"] = _spin_box(8, 1, 60, 1)
    win._ui["video_preset_combo"] = QtW.QComboBox()
    presets = [
        "Custom","512x512","640x360","768x432","960x540",
        "1024x576","1280x720","1920x1080",
    ]
    win._ui["video_preset_combo"].addItems(presets)

    video_row_container = QtW.QWidget()
    vlay = QtW.QHBoxLayout(video_row_container)
    vlay.setContentsMargins(0,0,0,0); vlay.setSpacing(6)
    vlay.addWidget(_labeled_group("Frames", win._ui["video_frames_spin"]))
    vlay.addWidget(_labeled_group("FPS", win._ui["video_fps_spin"]))
    vlay.addWidget(_labeled_group("Preset", win._ui["video_preset_combo"]))
    vlay.addStretch(1)
    win._ui["video_row_container"] = video_row_container

    form.addRow("Prompt:", win._ui["prompt_edit"])
    form.addRow("Negative:", win._ui["negative_edit"])
    dims_row = _hbox([
        _labeled_group("W", win._ui["width_spin"]),
        _labeled_group("H", win._ui["height_spin"]),
        _labeled_group("Steps", win._ui["steps_spin"]),
        _labeled_group("CFG", win._ui["cfg_spin"]),
    ])
    form.addRow("Dims/Steps:", dims_row)
    seed_row = _hbox([
        _labeled_group("Seed", win._ui["seed_spin"]),
        _labeled_group("Batch", win._ui["batch_spin"]),
        _labeled_group("Sampler", win._ui["sampler_combo"]),
    ])
    seed_auto_chk = QtW.QCheckBox("Auto Rand")
    win._ui["seed_auto_chk"] = seed_auto_chk
    seed_row.addWidget(seed_auto_chk)
    form.addRow("Seeds:", seed_row)
    form.addRow("Video:", video_row_container)

    # --- Upscaler row ---
    upscaler_row = _hbox([])
    upscaler_label = QtW.QLabel("Upscaler:")
    upscaler_combo = QtW.QComboBox()
    upscaler_combo.setMinimumWidth(220)
    win._ui["upscaler_combo"] = upscaler_combo
    upscaler_combo.addItem("None")
    btn_upscale = QtW.QPushButton("Upscale")
    btn_upscale.setEnabled(False)
    win._ui["btn_upscale"] = btn_upscale

    # --- Upload Image button ---
    btn_upload = QtW.QPushButton("Upload Image")
    win._ui["btn_upload"] = btn_upload

    upscaler_row.addWidget(upscaler_label)
    upscaler_row.addWidget(upscaler_combo)
    upscaler_row.addWidget(btn_upscale)
    upscaler_row.addWidget(btn_upload)
    upscaler_row.addStretch(1)
    layout.addLayout(upscaler_row)

    layout.addLayout(form)

    win._ui["preview_label"] = QtW.QLabel("Preview")
    win._ui["preview_label"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    win._ui["preview_label"].setMinimumHeight(240)
    win._ui["preview_label"].setFrameShape(QtW.QFrame.Shape.StyledPanel)
    layout.addWidget(win._ui["preview_label"], 1)

    btn_row = _hbox([])
    win._ui["btn_generate"] = QtW.QPushButton("Generate"); win._ui["btn_generate"].setVisible(False)
    win._ui["btn_cancel"] = QtW.QPushButton("Cancel"); win._ui["btn_cancel"].setEnabled(False)
    win._ui["btn_refine"] = QtW.QPushButton("Refine"); win._ui["btn_refine"].setEnabled(False)
    btn_row.addWidget(win._ui["btn_cancel"])
    btn_row.addWidget(win._ui["btn_refine"])

    # Anatomy Guard controls (additive, safe)
    anat_chk = QtW.QCheckBox("Anatomy Guard")
    anat_chk.setChecked(True)
    win._ui["anatomy_guard_chk"] = anat_chk
    btn_apply_anat = QtW.QPushButton("Apply Anatomy Guard")
    win._ui["btn_apply_anatomy_guard"] = btn_apply_anat
    btn_row.addWidget(anat_chk)
    btn_row.addWidget(btn_apply_anat)

    # Auto refine checkbox
    auto_refine_chk = QtW.QCheckBox("Auto refine")
    win._ui["auto_refine_chk"] = auto_refine_chk
    btn_row.addWidget(auto_refine_chk)

    btn_row.addStretch(1)
    layout.addLayout(btn_row)

    remember_chk = QtW.QCheckBox("Remember save location")
    win._ui["remember_save_chk"] = remember_chk
    btn_row.addWidget(remember_chk)
    cache_chk = QtW.QCheckBox("Use pipeline cache")
    win._ui["pipeline_cache_chk"] = cache_chk
    btn_row.addWidget(cache_chk)
    auto_gallery_chk = QtW.QCheckBox("Auto save to gallery")
    win._ui["auto_gallery_chk"] = auto_gallery_chk
    btn_row.addWidget(auto_gallery_chk)

    precision_label = QtW.QLabel("Precision:")
    layout.addWidget(precision_label)

    precision_combo = QtW.QComboBox()
    precision_combo.addItems(["FP16", "FP32"])
    precision_combo.setCurrentIndex(1)  # Default to FP32
    layout.addWidget(precision_combo)
    win._ui["precision_dropdown"] = precision_combo

    # New: Turbo decode precision
    turbo_dec_label = QtW.QLabel("Turbo Decode:")
    layout.addWidget(turbo_dec_label)

    turbo_decode_combo = QtW.QComboBox()
    turbo_decode_combo.addItems(["FP16", "FP32"])
    turbo_decode_combo.setCurrentIndex(1)  # Default to FP32
    layout.addWidget(turbo_decode_combo)
    win._ui["turbo_decode_dropdown"] = turbo_decode_combo

    win.setCentralWidget(central)
    if hasattr(win,"on_generate"):
        win._ui["btn_generate"].clicked.connect(win.on_generate)       # type: ignore
    if hasattr(win, "on_cancel_generate"):
        win._ui["btn_cancel"].clicked.connect(win.on_cancel_generate)  # type: ignore
    else:
        win._ui["btn_cancel"].setEnabled(False)
    if hasattr(win, "on_refine"):
        win._ui["btn_refine"].clicked.connect(win.on_refine)       # type: ignore
    else:
        win._ui["btn_refine"].setEnabled(False)
    if hasattr(win, "on_upscale"):
        win._ui["btn_upscale"].clicked.connect(win.on_upscale)    # type: ignore
    else:
        win._ui["btn_upscale"].setEnabled(False)
    if hasattr(win, "on_upload_image"):
        win._ui["btn_upload"].clicked.connect(win.on_upload_image)    # type: ignore
    else:
        win._ui["btn_upload"].setEnabled(True)
    # Anatomy Guard signal hookups (no-op if handlers absent)
    if hasattr(win, "on_apply_anatomy_guard"):
        win._ui["btn_apply_anatomy_guard"].clicked.connect(win.on_apply_anatomy_guard)  # type: ignore
    else:
        win._ui["btn_apply_anatomy_guard"].setEnabled(False)
    if hasattr(win, "on_toggle_anatomy_guard"):
        win._ui["anatomy_guard_chk"].toggled.connect(win.on_toggle_anatomy_guard)  # type: ignore

    # Video dimension / preset handlers (optional)
    if hasattr(win, "on_video_preset_changed"):
        win._ui["video_preset_combo"].currentTextChanged.connect(win.on_video_preset_changed)  # type: ignore
    if hasattr(win, "on_video_dims_manual_changed"):
        win._ui["width_spin"].valueChanged.connect(win.on_video_dims_manual_changed)   # type: ignore
        win._ui["height_spin"].valueChanged.connect(win.on_video_dims_manual_changed)  # type: ignore
    else:
        pass

def _build_docks(win, services: Services) -> None:
    dock_log = QtW.QDockWidget("Log", win)
    dock_log.setObjectName("LogDock")
    txt = QtW.QPlainTextEdit(); txt.setReadOnly(True)
    dock_log.setWidget(txt)
    win.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock_log)
    win._ui["dock_log"] = dock_log
    win._ui["log_edit"] = txt

def _build_statusbar(win, services: Services) -> None:
    sb = QtW.QStatusBar()
    win.setStatusBar(sb)
    win._ui["status_vram"] = QtW.QLabel("VRAM: -")
    win._ui["status_msg"] = QtW.QLabel("Ready")
    sb.addPermanentWidget(win._ui["status_vram"])
    sb.addWidget(win._ui["status_msg"], 1)

def _apply_styles(win, qss_path: Optional[str]) -> None:
    if not qss_path: return
    try:
        from pathlib import Path
        p = Path(qss_path)
        if p.exists():
            win.setStyleSheet(p.read_text(encoding="utf-8"))
    except Exception:
        pass

def _make_action(win, text: str, slot: Callable, shortcut: Optional[str],
                 tip: Optional[str], icon: Optional[str] = None):
    QActionClass = getattr(QtGui, "QAction", None) or getattr(QtW, "QAction")
    act = QActionClass(text, win)
    if icon: act.setIcon(QtGui.QIcon(icon))
    if shortcut: act.setShortcut(QtGui.QKeySequence(shortcut))
    if tip:
        act.setStatusTip(tip)
        act.setToolTip(f"{text} ({shortcut})" if shortcut else text)
    act.triggered.connect(slot)  # type: ignore
    win.addAction(act)
    return act

def _spin_box(value: int, minimum: int, maximum: int, step: int = 1) -> QtW.QSpinBox:
    sb = QtW.QSpinBox(); sb.setRange(minimum, maximum); sb.setValue(value); sb.setSingleStep(step); return sb

def _double_spin_box(value: float, minimum: float, maximum: float, step: float) -> QtW.QDoubleSpinBox:
    dsb = QtW.QDoubleSpinBox()
    dsb.setDecimals(2); dsb.setRange(minimum, maximum); dsb.setValue(value); dsb.setSingleStep(step)
    return dsb

def _hbox(widgets) -> QtW.QHBoxLayout:
    layout = QtW.QHBoxLayout()
    layout.setContentsMargins(0,0,0,0); layout.setSpacing(6)
    for w in widgets: layout.addWidget(w)
    return layout

def _labeled_group(label: str, widget: QtW.QWidget) -> QtW.QWidget:
    box = QtW.QWidget()
    v = QtW.QVBoxLayout(box); v.setContentsMargins(0, 0, 0, 0)
    lab = QtW.QLabel(label); lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
    v.addWidget(lab); v.addWidget(widget)
    return box
