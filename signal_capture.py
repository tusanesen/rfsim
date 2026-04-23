"""
Signal Capture & Spectrum Visualizer  —  with Monitoring Receiver Channels
Receives UDP chunks from signal_gen.py, shows a live overview spectrum + time-domain
strip, and supports up to 3 software-defined monitoring receiver tabs (WB-1, WB-2, NB-1).
"""

import socket
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import FS, N_FFT, CHUNK, F_LO, F_HI, UDP_HOST, UDP_PORT

plt.style.use("dark_background")

# ── Constants ──────────────────────────────────────────────────────────────────
PACKET_BYTES = CHUNK * 4
TD_SECS      = 0.010
TD_SAMPLES   = int(FS * TD_SECS)
INTERVAL_MS  = 33

N_FFT_WB = N_FFT        # 4096  -> ~122 Hz/bin
N_FFT_NB = 32_768       # 32768 -> ~15  Hz/bin  (8x finer; resolves AM sidebands)

_freqs_ov  = np.fft.rfftfreq(N_FFT_WB, 1 / FS)
_ov_mask   = (_freqs_ov >= F_LO) & (_freqs_ov <= F_HI)
OV_FREQS   = _freqs_ov[_ov_mask] / 1000.0   # kHz, precomputed

_WIN_WB = np.blackman(N_FFT_WB)
_WIN_NB = np.blackman(N_FFT_NB)

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#1e1e1e"
BG2      = "#252526"
BG3      = "#2d2d2d"
FG       = "#d4d4d4"
FG_DIM   = "#888888"
ACCENT   = "#0e639c"
ACCENT2  = "#1177bb"
ENTRY_BG = "#3c3c3c"

# Fixed receiver slot definitions
CHANNEL_DEFS = {
    "WB-1": {"ch_type": "WB", "color": "#00BFFF"},
    "WB-2": {"ch_type": "WB", "color": "#FFD700"},
    "NB-1": {"ch_type": "NB", "color": "#00FF9F"},
}


# ── UDP receiver thread ────────────────────────────────────────────────────────
def _receiver_thread(pkt_queue: queue.Queue, stop_event: threading.Event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.settimeout(0.5)
    print(f"Listening on {UDP_HOST}:{UDP_PORT} ...")
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(PACKET_BYTES)
            if len(data) == PACKET_BYTES:
                chunk = np.frombuffer(data, dtype=np.float32)
                if not pkt_queue.full():
                    pkt_queue.put_nowait(chunk)
        except socket.timeout:
            continue
    sock.close()


# ── Spectrum computation ───────────────────────────────────────────────────────
def _spectrum(ring: np.ndarray, window: np.ndarray, n_fft: int) -> np.ndarray:
    """Blackman-windowed rfft -> dBFS, full output (caller applies mask)."""
    X = np.fft.rfft(ring * window)
    return 20 * np.log10(np.abs(X) / n_fft + 1e-12)


# ── ChannelCanvas ──────────────────────────────────────────────────────────────
class ChannelCanvas(tk.Frame):
    """Embedded matplotlib figure for one monitoring channel."""

    def __init__(self, parent, ch_type: str, color: str):
        super().__init__(parent, bg=BG2)
        self._ch_type = ch_type
        self._color   = color
        self._mask    = None
        self._freqs   = None

        fig = plt.Figure(figsize=(9, 5), facecolor="#0d0d0d")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#111111")
        ax.set_xlabel("Frequency (kHz)", color="gray")
        ax.set_ylabel("Power (dBFS)", color="gray")
        ax.tick_params(colors="gray")
        ax.grid(True, color="#2a2a2a", linewidth=0.6)
        ax.set_ylim(-90, 10)
        ax.set_title("Set center freq and bandwidth, then Apply.",
                     color="#555555", fontsize=9)

        self._line,  = ax.plot([], [], color=color, linewidth=0.9)
        self._vline  = ax.axvline(0, color=color, alpha=0.35, linewidth=0.8, visible=False)
        self._ax     = ax

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas = canvas

    def setup(self, fc_hz: float, bw_hz: float):
        n_fft = N_FFT_NB if self._ch_type == "NB" else N_FFT_WB
        freqs = np.fft.rfftfreq(n_fft, 1 / FS)
        lo    = fc_hz - bw_hz / 2
        hi    = fc_hz + bw_hz / 2
        self._mask  = (freqs >= lo) & (freqs <= hi)
        self._freqs = freqs[self._mask] / 1000.0

        fc_k = fc_hz / 1000
        self._ax.set_xlim(lo / 1000, hi / 1000)
        self._vline.set_xdata([fc_k, fc_k])
        self._vline.set_visible(True)
        res = FS / n_fft
        self._ax.set_title(
            f"{self._ch_type}  fc={fc_k:.1f} kHz  BW={bw_hz/1000:.1f} kHz"
            f"  res~{res:.0f} Hz/bin",
            color=self._color, fontsize=9,
        )
        self._line.set_data([], [])
        self._canvas.draw_idle()

    def refresh(self, ring_wb: np.ndarray, ring_nb: np.ndarray):
        if self._mask is None:
            return
        if self._ch_type == "NB":
            mag = _spectrum(ring_nb, _WIN_NB, N_FFT_NB)
        else:
            mag = _spectrum(ring_wb, _WIN_WB, N_FFT_WB)
        self._line.set_data(self._freqs, mag[self._mask])
        self._canvas.draw_idle()


# ── ParamPanel ─────────────────────────────────────────────────────────────────
class ParamPanel(tk.Frame):
    """Left-side parameter form for a monitoring channel tab."""

    _BW_LIMITS = {"WB": (20_000, 200_000), "NB": (1_000, 20_000)}

    def __init__(self, parent, ch_type: str, on_apply):
        super().__init__(parent, bg=BG2, padx=14, pady=14, width=210)
        self.pack_propagate(False)
        self._ch_type  = ch_type
        self._on_apply = on_apply

        lo, hi  = self._BW_LIMITS[ch_type]
        bw_hint = f"{lo//1000} - {hi//1000} kHz"
        bw_def  = "60" if ch_type == "WB" else "10"
        hdr_col = "#00BFFF" if ch_type == "WB" else "#00FF9F"

        tk.Label(self, text=f"{ch_type} Receiver", bg=BG2, fg=hdr_col,
                 font=("Consolas", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        tk.Label(self, text="Center Freq (kHz)", bg=BG2, fg=FG,
                 font=("Consolas", 9)).grid(row=1, column=0, sticky="w", pady=4)
        self._fc_var = tk.StringVar(value="100")
        tk.Entry(self, textvariable=self._fc_var, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=("Consolas", 9),
                 width=10, relief="flat", bd=4).grid(
            row=1, column=1, sticky="w", padx=(8, 0))

        tk.Label(self, text=f"Bandwidth (kHz)\n({bw_hint})", bg=BG2, fg=FG,
                 font=("Consolas", 9), justify="left").grid(
            row=2, column=0, sticky="w", pady=4)
        self._bw_var = tk.StringVar(value=bw_def)
        tk.Entry(self, textvariable=self._bw_var, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=("Consolas", 9),
                 width=10, relief="flat", bd=4).grid(
            row=2, column=1, sticky="w", padx=(8, 0))

        tk.Frame(self, bg=BG3, height=1).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=10)

        tk.Button(self, text="Apply", command=self._apply,
                  bg=ACCENT, fg=FG, activebackground=ACCENT2, activeforeground="white",
                  relief="flat", font=("Consolas", 9), padx=10, pady=4,
                  cursor="hand2", bd=0).grid(row=4, column=0, columnspan=2, sticky="w")

        self._info = tk.Label(self, text="", bg=BG2, fg=FG_DIM,
                              font=("Consolas", 8), wraplength=185, justify="left")
        self._info.grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _apply(self):
        nyq = FS / 2
        try:
            fc_hz = float(self._fc_var.get()) * 1000
            if not (0 < fc_hz < nyq):
                messagebox.showerror("Input Error",
                                     f"Center freq must be between 0 and {nyq/1000:.0f} kHz.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Center freq must be a number.")
            return

        lo, hi = self._BW_LIMITS[self._ch_type]
        try:
            bw_hz = float(self._bw_var.get()) * 1000
            if not (lo <= bw_hz <= hi):
                messagebox.showerror(
                    "Input Error",
                    f"Bandwidth must be {lo//1000}–{hi//1000} kHz for {self._ch_type}.",
                )
                return
        except ValueError:
            messagebox.showerror("Input Error", "Bandwidth must be a number.")
            return

        n_fft = N_FFT_NB if self._ch_type == "NB" else N_FFT_WB
        res   = FS / n_fft
        self._info.config(
            text=f"fc = {fc_hz/1000:.1f} kHz\nBW = {bw_hz/1000:.1f} kHz\nres ~ {res:.0f} Hz/bin"
        )
        self._on_apply(fc_hz, bw_hz)


# ── ChannelTab ─────────────────────────────────────────────────────────────────
class ChannelTab(tk.Frame):
    def __init__(self, parent, name: str, ch_type: str, color: str, state: dict):
        super().__init__(parent, bg=BG)
        self.name    = name
        self._state  = state
        self._active = False

        params = ParamPanel(self, ch_type=ch_type, on_apply=self._on_apply)
        sep    = tk.Frame(self, bg=BG3, width=1)
        canvas = ChannelCanvas(self, ch_type=ch_type, color=color)

        params.pack(side="left", fill="y")
        sep.pack(side="left", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._canvas = canvas

    def _on_apply(self, fc_hz: float, bw_hz: float):
        self._canvas.setup(fc_hz, bw_hz)
        self._active = True

    def update(self):
        if self._active:
            self._canvas.refresh(self._state["ring_wb"], self._state["ring_nb"])


# ── OverviewTab ────────────────────────────────────────────────────────────────
class OverviewTab(tk.Frame):
    def __init__(self, parent, app, state: dict):
        super().__init__(parent, bg=BG)
        self._app        = app
        self._state      = state
        self._btn_active: dict[str, bool] = {n: False for n in CHANNEL_DEFS}
        self._btns:       dict[str, tk.Button] = {}
        self._got_signal = False
        self._build()

    def _build(self):
        fig = plt.Figure(figsize=(14, 7), facecolor="#0d0d0d")
        fig.suptitle("Signal Capture  -  Live Spectrum",
                     fontsize=13, color="white", y=0.99)

        gs    = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[7, 3],
                                  hspace=0.38, left=0.07, right=0.97, top=0.94, bottom=0.08)
        ax_sp = fig.add_subplot(gs[0])
        ax_td = fig.add_subplot(gs[1])

        ax_sp.set_facecolor("#111111")
        ax_sp.set_xlim(F_LO / 1000, F_HI / 1000)
        ax_sp.set_ylim(-90, 10)
        ax_sp.set_xlabel("Frequency (kHz)", color="gray")
        ax_sp.set_ylabel("Power (dBFS)", color="gray")
        ax_sp.tick_params(colors="gray")
        ax_sp.grid(True, color="#2a2a2a", linewidth=0.6)
        ax_sp.set_title("Waiting for signal...", color="#555555", fontsize=9)

        ax_td.set_facecolor("#0a0a0a")
        ax_td.set_xlim(0, TD_SECS * 1000)
        ax_td.set_ylim(-6, 6)
        ax_td.set_xlabel("Time (ms)", color="gray", fontsize=9)
        ax_td.set_ylabel("Amplitude", color="gray", fontsize=9)
        ax_td.tick_params(colors="gray", labelsize=7)
        ax_td.grid(True, color="#222222", linewidth=0.4)
        ax_td.set_title("Composite waveform (rolling 10 ms)", color="#888888", fontsize=8)

        t_ms = np.linspace(0, TD_SECS * 1000, TD_SAMPLES)
        self._spec_line, = ax_sp.plot([], [], color="#00BFFF", linewidth=0.9)
        self._td_line,   = ax_td.plot(t_ms, np.zeros(TD_SAMPLES),
                                       color="#00FF9F", linewidth=0.5)
        self._ax_sp = ax_sp

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas = canvas

        # ── channel activation button row ──────────────────────────
        bot = tk.Frame(self, bg=BG3, pady=6)
        bot.pack(fill="x", padx=10)
        tk.Label(bot, text="Monitoring Receivers:", bg=BG3, fg=FG_DIM,
                 font=("Consolas", 9)).pack(side="left", padx=(4, 12))
        for name, defs in CHANNEL_DEFS.items():
            color = defs["color"]
            btn = tk.Button(
                bot, text=name,
                bg=BG2, fg=color,
                activebackground=BG3, activeforeground=color,
                relief="flat", font=("Consolas", 9, "bold"),
                padx=12, pady=3, cursor="hand2", bd=0,
                command=lambda n=name: self._toggle(n),
            )
            btn.pack(side="left", padx=4)
            self._btns[name] = btn

    def _toggle(self, name: str):
        active = not self._btn_active[name]
        self._btn_active[name] = active
        defs  = CHANNEL_DEFS[name]
        color = defs["color"]
        btn   = self._btns[name]
        if active:
            btn.config(bg=color, fg=BG)
            self._app.activate_channel(name, defs["ch_type"], color)
        else:
            btn.config(bg=BG2, fg=color)
            self._app.deactivate_channel(name)

    def update(self, updated: bool):
        if not updated:
            return
        if not self._got_signal:
            self._ax_sp.set_title("", color="gray", fontsize=9)
            self._got_signal = True
        mag = _spectrum(self._state["ring_wb"], _WIN_WB, N_FFT_WB)
        self._spec_line.set_data(OV_FREQS, mag[_ov_mask])
        self._td_line.set_ydata(self._state["td_buffer"])
        self._canvas.draw_idle()


# ── ReceiverApp ────────────────────────────────────────────────────────────────
class ReceiverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signal Capture")
        self.configure(bg=BG)
        self.geometry("1200x720")

        self._pkt_queue  = queue.Queue(maxsize=400)
        self._stop_event = threading.Event()
        self._channels: dict[str, ChannelTab] = {}

        self._state = {
            "ring_wb":   np.zeros(N_FFT_WB),
            "ring_nb":   np.zeros(N_FFT_NB),
            "td_buffer": np.zeros(TD_SAMPLES),
        }

        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True)
        self._style_notebook()

        self._overview = OverviewTab(self._nb, app=self, state=self._state)
        self._nb.add(self._overview, text="  Overview  ")

        rx = threading.Thread(
            target=_receiver_thread,
            args=(self._pkt_queue, self._stop_event),
            daemon=True,
        )
        rx.start()

        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _style_notebook(self):
        s = ttk.Style()
        s.theme_use("default")
        s.configure("TNotebook",     background=BG3, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG2, foreground=FG_DIM,
                    font=("Consolas", 9), padding=[10, 4])
        s.map("TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", FG)])

    def activate_channel(self, name: str, ch_type: str, color: str):
        if name in self._channels:
            return
        tab = ChannelTab(self._nb, name=name, ch_type=ch_type,
                         color=color, state=self._state)
        self._channels[name] = tab
        self._nb.add(tab, text=f"  {name}  ")
        self._nb.select(tab)

    def deactivate_channel(self, name: str):
        if name not in self._channels:
            return
        tab = self._channels.pop(name)
        self._nb.forget(tab)
        tab.destroy()

    def _poll(self):
        updated = False
        while not self._pkt_queue.empty():
            try:
                chunk = self._pkt_queue.get_nowait()
            except queue.Empty:
                break
            s = self._state
            s["ring_wb"][:-CHUNK]   = s["ring_wb"][CHUNK:]
            s["ring_wb"][-CHUNK:]   = chunk
            s["ring_nb"][:-CHUNK]   = s["ring_nb"][CHUNK:]
            s["ring_nb"][-CHUNK:]   = chunk
            s["td_buffer"][:-CHUNK] = s["td_buffer"][CHUNK:]
            s["td_buffer"][-CHUNK:] = chunk
            updated = True

        self._overview.update(updated)
        for ch in self._channels.values():
            ch.update()

        self.after(INTERVAL_MS, self._poll)

    def _on_close(self):
        self._stop_event.set()
        self.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ReceiverApp().mainloop()
