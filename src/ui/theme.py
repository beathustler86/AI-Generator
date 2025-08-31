import tkinter as tk
from tkinter import ttk

PALETTE = {
    "bg_dark": "#2D2D30",
    "panel": "#3E3E42",
    "border": "#46464A",
    "surface": "#F3F3F3",
    "text": "#DCDCDC",
    "text_muted": "#9B9B9B",
    "accent": "#007ACC",
    "green": "#22C55E",
    "green_hover": "#16A34A",
    "green_active": "#15803D",
    "error": "#D13438",
    "warn": "#F2CC0C",
    "success": "#107C10",
}

def apply_theme(root: tk.Misc):
    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass

    # Base frames / labels
    style.configure("App.TFrame", background=PALETTE["bg_dark"])
    style.configure("Panel.TFrame", background=PALETTE["panel"])
    style.configure("Nav.TFrame", background=PALETTE["panel"])
    style.configure("TLabel", background=PALETTE["panel"], foreground=PALETTE["text"], font=("Segoe UI", 10))
    style.configure("Muted.TLabel", background=PALETTE["panel"], foreground=PALETTE["text_muted"], font=("Segoe UI", 9))

    # Buttons
    style.configure(
        "Primary.TButton",
        font=("Segoe UI Semibold", 10),
        foreground="white",
        background=PALETTE["green"],
        padding=(14, 6),
        borderwidth=0,
        focuscolor=PALETTE["accent"]
    )
    style.map(
        "Primary.TButton",
        background=[("active", PALETTE["green_hover"]), ("pressed", PALETTE["green_active"])],
        foreground=[("disabled", PALETTE["text_muted"])]
    )
    style.configure(
        "Accent.TButton",
        font=("Segoe UI", 10),
        foreground="white",
        background=PALETTE["accent"],
        padding=(10, 4),
        borderwidth=0
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#0064B5"), ("pressed", "#005A9A")]
    )
    style.configure("Flat.TButton", font=("Segoe UI", 9), background=PALETTE["panel"], foreground=PALETTE["text_muted"], relief="flat", padding=(8,4))
    style.map("Flat.TButton", background=[("active", PALETTE["bg_dark"])], foreground=[("active", PALETTE["text"])])

    # Progress bars
    style.configure("Thin.Horizontal.TProgressbar",
                    troughcolor=PALETTE["bg_dark"],
                    background=PALETTE["accent"],
                    darkcolor=PALETTE["accent"],
                    lightcolor=PALETTE["accent"],
                    bordercolor=PALETTE["bg_dark"],
                    thickness=6)

    # Telemetry mini bars
    style.configure("Metric.Horizontal.TProgressbar",
                    troughcolor=PALETTE["bg_dark"],
                    background=PALETTE["green"],
                    thickness=10)

    # Scales
    style.configure("TScale",
                    background=PALETTE["panel"],
                    troughcolor=PALETTE["bg_dark"],
                    sliderlength=18)
    style.map("TScale",
              background=[("active", PALETTE["panel"])],
              troughcolor=[("active", PALETTE["bg_dark"])])

    # Status bar
    style.configure("Status.TLabel", background=PALETTE["panel"], foreground=PALETTE["text_muted"], font=("Segoe UI", 9))

    # Notebook
    style.configure("Workspace.TNotebook", background=PALETTE["panel"], borderwidth=0)
    style.configure("Workspace.TNotebook.Tab", background=PALETTE["bg_dark"], foreground=PALETTE["text"], padding=(12, 6))
    style.map("Workspace.TNotebook.Tab",
              background=[("selected", PALETTE["panel"])],
              foreground=[("disabled", PALETTE["text_muted"])])

    return style