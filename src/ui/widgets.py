import tkinter as tk
from tkinter import ttk
import time

class Tooltip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._cancel)

    def _schedule(self, _):
        self._id = self.widget.after(self.delay, self._show)

    def _cancel(self, _):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None

    def _show(self):
        if self._tip:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.configure(bg="#46464A")
        lbl = tk.Label(self._tip, text=self.text, bg="#46464A", fg="#DCDCDC", font=("Segoe UI", 9))
        lbl.pack(ipadx=6, ipady=3)
        self._tip.wm_geometry(f"+{x}+{y}")

class LabeledSlider(ttk.Frame):
    def __init__(self, master, text, from_, to, variable, resolution=1, width=180, tooltip=None):
        super().__init__(master, style="Panel.TFrame")
        self.label = ttk.Label(self, text=text)
        self.label.grid(row=0, column=0, sticky="w")
        self.value_label = ttk.Label(self, text=str(variable.get()), style="Muted.TLabel")
        self.value_label.grid(row=0, column=1, sticky="e", padx=(6,0))
        self.scale = ttk.Scale(self, from_=from_, to=to, orient="horizontal",
                               variable=variable, length=width, command=self._on_move)
        self.scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.columnconfigure(0, weight=1)
        if tooltip:
            Tooltip(self.scale, tooltip)

    def _on_move(self, _event):
        self.value_label.config(text=f"{self.scale.get():.2f}" if "." in str(self.scale.get()) else str(int(self.scale.get())))

class MetricBar(ttk.Frame):
    def __init__(self, master, label, max_value=100, bar_style="Metric.Horizontal.TProgressbar", fg="#DCDCDC"):
        super().__init__(master, style="Panel.TFrame")
        self.var = tk.DoubleVar(value=0.0)
        ttk.Label(self, text=label, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.pb = ttk.Progressbar(self, orient="horizontal", style=bar_style, variable=self.var, maximum=max_value, length=160, mode="determinate")
        self.pb.grid(row=1, column=0, sticky="ew")
        self.text = ttk.Label(self, text="0%", style="Muted.TLabel")
        self.text.grid(row=1, column=1, padx=(6,0))
        self.columnconfigure(0, weight=1)

    def update_value(self, val, suffix="%"):
        self.var.set(val)
        self.text.config(text=f"{val:.0f}{suffix}")

class ToastManager:
    def __init__(self, root):
        self.root = root
        self.toasts = []

    def show(self, message, kind="info", duration=3000):
        win = tk.Toplevel(self.root)
        win.wm_overrideredirect(True)
        win.configure(bg="#2D2D30")
        color = {"info":"#007ACC","success":"#107C10","warn":"#F2CC0C","error":"#D13438"}.get(kind,"#007ACC")
        frame = tk.Frame(win, bg="#3E3E42", bd=1, relief="solid", highlightbackground=color, highlightcolor=color, highlightthickness=1)
        frame.pack()
        tk.Label(frame, text=message, font=("Segoe UI", 9), bg="#3E3E42", fg="#DCDCDC").pack(padx=12, pady=8)
        self.toasts.append(win)
        self._reposition()
        win.after(duration, lambda: self._close(win))

    def _reposition(self):
        # stack top-right
        sw = self.root.winfo_screenwidth()
        y = 40
        for win in self.toasts:
            win.update_idletasks()
            w = win.winfo_width()
            win.geometry(f"{w}x{win.winfo_height()}+{sw - w - 24}+{y}")
            y += win.winfo_height() + 6

    def _close(self, win):
        try:
            self.toasts.remove(win)
        except ValueError:
            pass
        win.destroy()
        self._reposition()

def pulse_primary(style: ttk.Style, base="Primary.TButton", colors=None, steps=6, interval=120, i=0):
    colors = colors or ["#22C55E", "#16A34A", "#22C55E"]
    if i >= steps:
        style.configure(base, background=colors[0])
        return
    style.configure(base, background=colors[i % len(colors)])
    style.master.after(interval, pulse_primary, style, base, colors, steps, interval, i+1)