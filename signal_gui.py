"""
Signal Generator GUI.
App receives shared state from signal_gen.py and owns only the tkinter layer.
Signal model knowledge comes entirely through SignalBase / SIGNAL_REGISTRY.
"""

import time
import tkinter as tk
from tkinter import ttk, messagebox

from config import NOISE_STD, UDP_HOST, UDP_PORT
from signals import SignalBase, SIGNAL_REGISTRY

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#1e1e1e"
BG2      = "#252526"
BG3      = "#2d2d2d"
FG       = "#d4d4d4"
FG_DIM   = "#888888"
ACCENT   = "#0e639c"
ACCENT2  = "#1177bb"
ENTRY_BG = "#3c3c3c"
SEL_BG   = "#094771"

# Listbox colour per mod_type key
_TYPE_COLOR = {
    "AM":  FG,
    "FM":  "#FFD700",
    "ASK": "#00FF9F",
}


class App(tk.Tk):
    def __init__(self, signals: list, sig_lock, noise_ref: list,
                 pkt_counter: list, stop_evt):
        super().__init__()
        self.title("Signal Generator")
        self.configure(bg=BG)
        self.resizable(True, False)
        self.minsize(820, 440)

        # shared state owned by signal_gen.py
        self.signals     = signals
        self.sig_lock    = sig_lock
        self.noise_ref   = noise_ref
        self.pkt_counter = pkt_counter
        self.stop_evt    = stop_evt

        self._sel_idx  = None
        self._rate_ts  = time.perf_counter()
        self._type_entries: dict[str, tk.Entry] = {}

        self._build_ui()
        self._refresh_list()
        self._tick()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG, pady=6)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Signal Generator", bg=BG, fg="#00BFFF",
                 font=("Consolas", 13, "bold")).pack(side="left")
        self._status_lbl = tk.Label(hdr, text="", bg=BG, fg="#00FF9F",
                                    font=("Consolas", 9))
        self._status_lbl.pack(side="right", padx=12)

        tk.Frame(self, bg=BG3, height=1).pack(fill="x")

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        self._build_list_panel(body)
        self._build_form_panel(body)

        # Bottom bar
        tk.Frame(self, bg=BG3, height=1).pack(fill="x")
        bot = tk.Frame(self, bg=BG3, pady=5)
        bot.pack(fill="x", padx=10)
        tk.Label(bot, text="Noise Floor:", bg=BG3, fg=FG_DIM,
                 font=("Consolas", 9)).pack(side="left", padx=(4, 4))
        self._noise_var = tk.DoubleVar(value=NOISE_STD)
        tk.Scale(bot, from_=0.0, to=0.10, resolution=0.001, orient="horizontal",
                 variable=self._noise_var, command=self._on_noise,
                 bg=BG3, fg=FG, troughcolor=ENTRY_BG, highlightthickness=0,
                 activebackground=ACCENT2, font=("Consolas", 8),
                 length=180, showvalue=True).pack(side="left")
        self._rate_lbl = tk.Label(bot, text="TX: 0 pkt/s", bg=BG3, fg=FG_DIM,
                                  font=("Consolas", 9))
        self._rate_lbl.pack(side="right", padx=8)

    def _build_list_panel(self, parent):
        left = tk.Frame(parent, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(left, text="Active Signals", bg=BG, fg=FG_DIM,
                 font=("Consolas", 9)).pack(anchor="w", pady=(0, 3))

        wrap = tk.Frame(left, bg=ENTRY_BG)
        wrap.pack(fill="both", expand=True)
        self._lb = tk.Listbox(
            wrap, bg=BG2, fg=FG, selectbackground=SEL_BG, selectforeground="white",
            font=("Consolas", 9), relief="flat", bd=0, width=56, height=12,
            activestyle="none",
        )
        sb = tk.Scrollbar(wrap, orient="vertical", command=self._lb.yview,
                          bg=BG3, troughcolor=BG2, width=10)
        self._lb.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._lb.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._lb.bind("<<ListboxSelect>>", self._on_select)

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(5, 0))
        self._mkbtn(btn_row, "Remove",        self._remove_signal).pack(side="left", padx=(0, 4))
        self._mkbtn(btn_row, "Toggle On/Off", self._toggle_signal).pack(side="left")

    def _build_form_panel(self, parent):
        right = tk.Frame(parent, bg=BG2, padx=12, pady=10)
        right.pack(side="left", fill="y")

        tk.Label(right, text="Configure Signal", bg=BG2, fg="#00BFFF",
                 font=("Consolas", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # Mod type selector — built dynamically from SIGNAL_REGISTRY
        tk.Label(right, text="Modulation", bg=BG2, fg=FG,
                 font=("Consolas", 9)).grid(row=1, column=0, sticky="w", pady=3)
        type_frame = tk.Frame(right, bg=BG2)
        type_frame.grid(row=1, column=1, sticky="w")
        self._mod_type = tk.StringVar(value=next(iter(SIGNAL_REGISTRY)))
        for mod in SIGNAL_REGISTRY:
            tk.Radiobutton(
                type_frame, text=mod, variable=self._mod_type, value=mod,
                bg=BG2, fg=_TYPE_COLOR.get(mod, FG), selectcolor=BG3,
                activebackground=BG2, activeforeground=FG,
                font=("Consolas", 9, "bold"),
                command=self._on_type_change,
            ).pack(side="left", padx=(0, 8))

        # Common fields (always visible)
        self._common_entries: dict[str, tk.Entry] = {}
        cls0 = SIGNAL_REGISTRY[self._mod_type.get()]
        for row_i, spec in enumerate(cls0.common_field_specs(), start=2):
            tk.Label(right, text=spec.label, bg=BG2, fg=FG,
                     font=("Consolas", 9)).grid(row=row_i, column=0, sticky="w", pady=3)
            e = tk.Entry(right, bg=ENTRY_BG, fg=FG, insertbackground=FG,
                         font=("Consolas", 9), width=14, relief="flat", bd=4)
            e.insert(0, spec.default)
            e.grid(row=row_i, column=1, sticky="w", padx=(8, 0), pady=3)
            self._common_entries[spec.key] = e

        # Separator
        tk.Frame(right, bg=BG3, height=1).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=6)

        # Type-specific fields (rebuilt on type change)
        self._type_frame = tk.Frame(right, bg=BG2)
        self._type_frame.grid(row=6, column=0, columnspan=2, sticky="ew")
        self._rebuild_type_fields()

        # Action buttons
        btn_row = tk.Frame(right, bg=BG2)
        btn_row.grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._mkbtn(btn_row, "Add Signal",      self._add_signal,    accent=True).pack(side="left", padx=(0, 6))
        self._mkbtn(btn_row, "Update Selected", self._update_signal).pack(side="left")

    def _mkbtn(self, parent, text, cmd, accent=False):
        return tk.Button(
            parent, text=text, command=cmd,
            bg=ACCENT if accent else BG3, fg=FG,
            activebackground=ACCENT2, activeforeground="white",
            relief="flat", font=("Consolas", 9), padx=8, pady=3,
            cursor="hand2", bd=0,
        )

    # ── Dynamic type-specific field area ─────────────────────────────────────
    def _rebuild_type_fields(self, prefill: dict = None):
        for w in self._type_frame.winfo_children():
            w.destroy()
        self._type_entries.clear()

        cls = SIGNAL_REGISTRY[self._mod_type.get()]
        for i, spec in enumerate(cls.type_field_specs()):
            tk.Label(self._type_frame, text=spec.label, bg=BG2, fg=FG,
                     font=("Consolas", 9)).grid(row=i, column=0, sticky="w", pady=3)

            val = prefill.get(spec.key, spec.default) if prefill else spec.default

            if spec.choices is not None:
                # Style the combobox to match the dark theme
                style = ttk.Style()
                style.theme_use("default")
                style.configure("Dark.TCombobox",
                                fieldbackground=ENTRY_BG, background=BG3,
                                foreground=FG, selectbackground=SEL_BG,
                                selectforeground="white", arrowcolor=FG)
                e = ttk.Combobox(self._type_frame, values=spec.choices,
                                 font=("Consolas", 9), width=20,
                                 state="readonly", style="Dark.TCombobox")
                e.set(val if val in spec.choices else (spec.choices[0] if spec.choices else ""))
            else:
                e = tk.Entry(self._type_frame, bg=ENTRY_BG, fg=FG,
                             insertbackground=FG, font=("Consolas", 9),
                             width=22, relief="flat", bd=4)
                e.insert(0, val)

            e.grid(row=i, column=1, sticky="w", padx=(8, 0), pady=3)
            self._type_entries[spec.key] = e

    def _on_type_change(self):
        self._rebuild_type_fields()  # resets type fields to defaults

    # ── List helpers ──────────────────────────────────────────────────────────
    def _refresh_list(self):
        self._lb.delete(0, tk.END)
        with self.sig_lock:
            for s in self.signals:
                self._lb.insert(tk.END, s.describe())
                color = FG_DIM if not s.enabled else _TYPE_COLOR.get(s.mod_type, FG)
                self._lb.itemconfig(tk.END, fg=color)
        if self._sel_idx is not None and self._sel_idx < self._lb.size():
            self._lb.selection_set(self._sel_idx)

    def _on_select(self, _=None):
        sel = self._lb.curselection()
        if not sel:
            return
        self._sel_idx = sel[0]
        with self.sig_lock:
            s = self.signals[self._sel_idx]

        form = s.to_form()
        self._mod_type.set(s.mod_type)
        self._rebuild_type_fields(prefill=form)  # type fields with signal's values

        for key, entry in self._common_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, form.get(key, ""))

    # ── CRUD ──────────────────────────────────────────────────────────────────
    def _collect_form(self) -> dict[str, str]:
        values = {k: e.get() for k, e in self._common_entries.items()}
        values.update({k: e.get() for k, e in self._type_entries.items()})
        return values

    def _parse_and_validate(self) -> "SignalBase | None":
        cls    = SIGNAL_REGISTRY[self._mod_type.get()]
        values = self._collect_form()
        errors = cls.validate(values)
        if errors:
            messagebox.showerror("Input Error", "\n".join(errors))
            return None
        return cls.from_form(values)

    def _add_signal(self):
        s = self._parse_and_validate()
        if s is None:
            return
        with self.sig_lock:
            self.signals.append(s)
        self._sel_idx = len(self.signals) - 1
        self._refresh_list()
        self._lb.see(tk.END)

    def _update_signal(self):
        if self._sel_idx is None:
            messagebox.showinfo("No Selection", "Select a signal in the list first.")
            return
        s = self._parse_and_validate()
        if s is None:
            return
        with self.sig_lock:
            s.enabled = self.signals[self._sel_idx].enabled
            self.signals[self._sel_idx] = s
        self._refresh_list()

    def _remove_signal(self):
        if self._sel_idx is None:
            return
        with self.sig_lock:
            self.signals.pop(self._sel_idx)
        self._sel_idx = None
        self._refresh_list()

    def _toggle_signal(self):
        if self._sel_idx is None:
            return
        with self.sig_lock:
            self.signals[self._sel_idx].enabled = not self.signals[self._sel_idx].enabled
        self._refresh_list()

    # ── Noise & status ────────────────────────────────────────────────────────
    def _on_noise(self, _=None):
        self.noise_ref[0] = self._noise_var.get()

    def _tick(self):
        now     = time.perf_counter()
        elapsed = now - self._rate_ts
        if elapsed >= 1.0:
            rate = self.pkt_counter[0] / elapsed
            self.pkt_counter[0] = 0
            self._rate_ts = now
            self._rate_lbl.config(text=f"TX: {rate:.0f} pkt/s")

        with self.sig_lock:
            enabled = sum(1 for s in self.signals if s.enabled)
            total   = len(self.signals)
        self._status_lbl.config(
            text=f"-> {UDP_HOST}:{UDP_PORT}   {enabled}/{total} signals active"
        )
        self.after(500, self._tick)

    def _on_close(self):
        self.stop_evt.set()
        self.destroy()
