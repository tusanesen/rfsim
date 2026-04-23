"""
Signal Generator — AM/FM transmitter with live GUI control
Streams composite signal over UDP to signal_capture.py

Run this first, then start signal_capture.py.
"""

import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox
from dataclasses import dataclass
import numpy as np

from config import FS, CHUNK, NOISE_STD, UDP_HOST, UDP_PORT

PACKET_BYTES = CHUNK * 4
TWO_PI       = 2 * np.pi

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


# ── Signal model ──────────────────────────────────────────────────────────────
@dataclass
class Signal:
    mod_type:  str    # "AM" or "FM"
    fc:        float  # carrier (Hz)
    fm:        float  # modulating frequency (Hz)
    mod_param: float  # AM → modulation index (0–1) | FM → deviation (Hz)
    amplitude: float  # linear amplitude
    label:     str
    enabled:   bool = True

    def describe(self) -> str:
        if self.mod_type == "AM":
            p = f"m={self.mod_param:.2f}"
        else:
            p = f"dev={self.mod_param/1000:.1f}kHz"
        dot = "[on] " if self.enabled else "[off]"
        return (f"{dot} [{self.mod_type}]  {self.label:<6}  "
                f"fc={self.fc/1000:.1f}kHz  fm={self.fm/1000:.2f}kHz  "
                f"{p}  A={self.amplitude:.2f}")


def _default_signals() -> list:
    return [
        Signal("AM",  60_000,  1_000, 0.50, 1.0, "CH1"),
        Signal("AM",  90_000,  2_500, 0.70, 1.0, "CH2"),
        Signal("AM", 130_000,  5_000, 0.40, 1.0, "CH3"),
        Signal("FM", 170_000, 10_000, 8_000, 1.0, "CH4"),  # FM: 8 kHz deviation
        Signal("AM", 210_000,  3_000, 0.60, 1.0, "CH5"),
    ]


# ── Signal generation ─────────────────────────────────────────────────────────
def generate_chunk(t_offset: float, sigs: list, noise_std: float) -> np.ndarray:
    t   = np.arange(CHUNK, dtype=np.float64) / FS + t_offset
    out = np.zeros(CHUNK, dtype=np.float64)
    for s in sigs:
        if not s.enabled:
            continue
        if s.mod_type == "AM":
            out += s.amplitude * (1.0 + s.mod_param * np.cos(TWO_PI * s.fm * t)) \
                               * np.cos(TWO_PI * s.fc * t)
        else:  # FM
            beta = s.mod_param / s.fm   # modulation index = deviation / fm
            out += s.amplitude * np.cos(TWO_PI * s.fc * t + beta * np.sin(TWO_PI * s.fm * t))
    out += np.random.randn(CHUNK) * noise_std
    return out.astype(np.float32)


# ── Transmitter thread ────────────────────────────────────────────────────────
def _tx_worker(stop_evt, signals, sig_lock, noise_ref, pkt_counter):
    sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest   = (UDP_HOST, UDP_PORT)
    t_off  = 0.0
    t_next = time.perf_counter()

    while not stop_evt.is_set():
        with sig_lock:
            snap = list(signals)

        chunk = generate_chunk(t_off, snap, noise_ref[0])
        t_off += CHUNK / FS

        try:
            sock.sendto(chunk.tobytes(), dest)
            pkt_counter[0] += 1
        except OSError:
            pass

        t_next += CHUNK / FS
        sleep_s = t_next - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)

    sock.close()


# ── GUI ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signal Generator")
        self.configure(bg=BG)
        self.resizable(True, False)
        self.minsize(780, 420)

        self.signals     = _default_signals()
        self.sig_lock    = threading.Lock()
        self.noise_ref   = [NOISE_STD]
        self.stop_evt    = threading.Event()
        self.pkt_counter = [0]
        self._rate_ts    = time.perf_counter()
        self._sel_idx    = None   # persists listbox selection across focus changes

        self._build_ui()
        self._refresh_list()

        self._tx = threading.Thread(
            target=_tx_worker,
            args=(self.stop_evt, self.signals, self.sig_lock,
                  self.noise_ref, self.pkt_counter),
            daemon=True,
        )
        self._tx.start()

        self._tick()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=BG, pady=6)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Signal Generator",
                 bg=BG, fg="#00BFFF", font=("Consolas", 13, "bold")).pack(side="left")
        self._status_lbl = tk.Label(hdr, text="", bg=BG, fg="#00FF9F",
                                    font=("Consolas", 9))
        self._status_lbl.pack(side="right", padx=12)

        sep = tk.Frame(self, bg=BG3, height=1)
        sep.pack(fill="x")

        # ── Body ──
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Left: signal list
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(left, text="Active Signals", bg=BG, fg=FG_DIM,
                 font=("Consolas", 9)).pack(anchor="w", pady=(0, 3))

        lb_wrap = tk.Frame(left, bg=ENTRY_BG)
        lb_wrap.pack(fill="both", expand=True)

        self._lb = tk.Listbox(
            lb_wrap, bg=BG2, fg=FG, selectbackground=SEL_BG, selectforeground="white",
            font=("Consolas", 9), relief="flat", bd=0, width=52, height=11,
            activestyle="none",
        )
        sb = tk.Scrollbar(lb_wrap, orient="vertical", command=self._lb.yview,
                          bg=BG3, troughcolor=BG2, width=10)
        self._lb.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._lb.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._lb.bind("<<ListboxSelect>>", self._on_select)

        btn_r = tk.Frame(left, bg=BG)
        btn_r.pack(fill="x", pady=(5, 0))
        self._mkbtn(btn_r, "Remove",        self._remove_signal).pack(side="left", padx=(0, 4))
        self._mkbtn(btn_r, "Toggle On/Off", self._toggle_signal).pack(side="left")

        # Right: form
        right = tk.Frame(body, bg=BG2, padx=12, pady=10)
        right.pack(side="left", fill="y")

        tk.Label(right, text="Configure Signal", bg=BG2, fg="#00BFFF",
                 font=("Consolas", 10, "bold")).grid(row=0, column=0, columnspan=2,
                                                      sticky="w", pady=(0, 8))

        # Modulation type
        tk.Label(right, text="Modulation", bg=BG2, fg=FG,
                 font=("Consolas", 9)).grid(row=1, column=0, sticky="w", pady=3)
        type_frame = tk.Frame(right, bg=BG2)
        type_frame.grid(row=1, column=1, sticky="w")
        self._mod_type = tk.StringVar(value="AM")
        for t in ("AM", "FM"):
            tk.Radiobutton(type_frame, text=t, variable=self._mod_type, value=t,
                           bg=BG2, fg=FG, selectcolor=BG3,
                           activebackground=BG2, activeforeground=FG,
                           font=("Consolas", 9),
                           command=self._on_type_change).pack(side="left", padx=(0, 8))

        # Text fields
        self._param_lbl_var = tk.StringVar(value="Mod Index  (0-1)")
        field_defs = [
            ("Label",                None,               "CH6"),
            ("Carrier  (kHz)",       None,               "100"),
            ("Mod Freq (kHz)",       None,               "2"),
            ("",                     self._param_lbl_var, "0.5"),
            ("Amplitude",            None,               "1.0"),
        ]
        self._entries = {}
        entry_keys = ["label", "fc", "fm", "param", "amp"]

        for i, (lbl_text, lbl_var, default) in enumerate(field_defs, start=2):
            if lbl_var:
                tk.Label(right, textvariable=lbl_var, bg=BG2, fg=FG,
                         font=("Consolas", 9)).grid(row=i, column=0, sticky="w", pady=3)
            else:
                tk.Label(right, text=lbl_text, bg=BG2, fg=FG,
                         font=("Consolas", 9)).grid(row=i, column=0, sticky="w", pady=3)
            e = tk.Entry(right, bg=ENTRY_BG, fg=FG, insertbackground=FG,
                         font=("Consolas", 9), width=14, relief="flat", bd=4)
            e.insert(0, default)
            e.grid(row=i, column=1, sticky="w", padx=(8, 0), pady=3)
            self._entries[entry_keys[i - 2]] = e

        # Action buttons
        btn_row = tk.Frame(right, bg=BG2)
        btn_row.grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._mkbtn(btn_row, "Add Signal",       self._add_signal,    accent=True).pack(side="left", padx=(0, 6))
        self._mkbtn(btn_row, "Update Selected",  self._update_signal).pack(side="left")

        # ── Bottom bar ──
        sep2 = tk.Frame(self, bg=BG3, height=1)
        sep2.pack(fill="x")
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

        self._rate_lbl = tk.Label(bot, text="TX: 0 pkt/s",
                                  bg=BG3, fg=FG_DIM, font=("Consolas", 9))
        self._rate_lbl.pack(side="right", padx=8)

    def _mkbtn(self, parent, text, cmd, accent=False):
        return tk.Button(parent, text=text, command=cmd,
                         bg=ACCENT if accent else BG3, fg=FG,
                         activebackground=ACCENT2, activeforeground="white",
                         relief="flat", font=("Consolas", 9),
                         padx=8, pady=3, cursor="hand2", bd=0)

    # ── List helpers ──────────────────────────────────────────────────────────
    def _refresh_list(self):
        self._lb.delete(0, tk.END)
        with self.sig_lock:
            for s in self.signals:
                self._lb.insert(tk.END, s.describe())
                if not s.enabled:
                    self._lb.itemconfig(tk.END, fg="#555555")
                elif s.mod_type == "FM":
                    self._lb.itemconfig(tk.END, fg="#FFD700")
        if self._sel_idx is not None and self._sel_idx < self._lb.size():
            self._lb.selection_set(self._sel_idx)

    def _on_select(self, _=None):
        sel = self._lb.curselection()
        if not sel:
            return
        self._sel_idx = sel[0]
        with self.sig_lock:
            s = self.signals[self._sel_idx]
        self._mod_type.set(s.mod_type)
        self._on_type_change()
        self._entries["label"].delete(0, tk.END);  self._entries["label"].insert(0, s.label)
        self._entries["fc"].delete(0, tk.END);     self._entries["fc"].insert(0, f"{s.fc/1000:.3f}")
        self._entries["fm"].delete(0, tk.END);     self._entries["fm"].insert(0, f"{s.fm/1000:.4f}")
        self._entries["amp"].delete(0, tk.END);    self._entries["amp"].insert(0, f"{s.amplitude:.2f}")
        self._entries["param"].delete(0, tk.END)
        if s.mod_type == "AM":
            self._entries["param"].insert(0, f"{s.mod_param:.3f}")
        else:
            self._entries["param"].insert(0, f"{s.mod_param:.1f}")

    def _on_type_change(self):
        if self._mod_type.get() == "AM":
            self._param_lbl_var.set("Mod Index  (0-1)")
        else:
            self._param_lbl_var.set("Deviation  (Hz)")

    # ── CRUD ──────────────────────────────────────────────────────────────────
    def _parse_form(self) -> "Signal | None":
        try:
            label     = self._entries["label"].get().strip() or "SIG"
            fc        = float(self._entries["fc"].get()) * 1000
            fm        = float(self._entries["fm"].get()) * 1000
            mod_param = float(self._entries["param"].get())
            amp       = float(self._entries["amp"].get())
            mod_type  = self._mod_type.get()
        except ValueError:
            messagebox.showerror("Input Error", "All numeric fields must be valid numbers.")
            return None

        nyq = FS / 2
        errors = []
        if not (0 < fc < nyq):
            errors.append(f"Carrier must be between 0 and {nyq/1000:.0f} kHz.")
        if fm <= 0:
            errors.append("Mod frequency must be > 0.")
        if mod_type == "AM" and not (0 < mod_param <= 1.0):
            errors.append("AM modulation index must be in (0, 1].")
        if mod_type == "FM" and mod_param <= 0:
            errors.append("FM deviation must be > 0 Hz.")
        if amp <= 0:
            errors.append("Amplitude must be > 0.")
        if errors:
            messagebox.showerror("Input Error", "\n".join(errors))
            return None

        return Signal(mod_type, fc, fm, mod_param, amp, label)

    def _add_signal(self):
        s = self._parse_form()
        if s is None:
            return
        with self.sig_lock:
            self.signals.append(s)
        self._refresh_list()
        self._lb.see(tk.END)

    def _update_signal(self):
        if self._sel_idx is None:
            messagebox.showinfo("No Selection", "Select a signal in the list first.")
            return
        s = self._parse_form()
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

    # ── Noise & status tick ───────────────────────────────────────────────────
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
            text=f"→ {UDP_HOST}:{UDP_PORT}   {enabled}/{total} signals active"
        )
        self.after(500, self._tick)

    def _on_close(self):
        self.stop_evt.set()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
