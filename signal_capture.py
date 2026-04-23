"""
Signal Capture & Spectrum Visualizer  —  with Monitoring Receiver Channels + DDR
Receives UDP chunks from signal_gen.py. Provides:
  - Overview tab: live spectrum + time-domain strip
  - WB/NB channel tabs: per-channel spectrum view
  - DDR sub-channels (WB only): DDC -> Demodulator -> Speaker pipeline
"""

import collections
import datetime
import os
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
import sounddevice as sd

from config import FS, N_FFT, CHUNK, F_LO, F_HI, UDP_HOST, UDP_PORT

RECORDINGS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
_MAX_REC_BYTES  = 10 * 1024 ** 3   # 10 GB


def _ensure_recordings_dir():
    """Create recordings folder and evict oldest files if total size > 10 GB."""
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    entries = []
    for name in os.listdir(RECORDINGS_DIR):
        path = os.path.join(RECORDINGS_DIR, name)
        if os.path.isfile(path):
            entries.append((os.path.getmtime(path), os.path.getsize(path), path))
    entries.sort()                          # oldest first
    total = sum(sz for _, sz, _ in entries)
    while total > _MAX_REC_BYTES and entries:
        mtime, sz, path = entries.pop(0)
        try:
            os.remove(path)
            total -= sz
            print(f"Evicted old recording: {os.path.basename(path)}")
        except OSError as e:
            print(f"Could not remove {path}: {e}")

plt.style.use("dark_background")

# ── Constants ──────────────────────────────────────────────────────────────────
PACKET_BYTES = CHUNK * 4
TD_SECS      = 0.010
TD_SAMPLES   = int(FS * TD_SECS)
INTERVAL_MS  = 33

N_FFT_WB = N_FFT        # 4096  -> ~122 Hz/bin
N_FFT_NB = 32_768       # 32768 -> ~15  Hz/bin

_freqs_ov  = np.fft.rfftfreq(N_FFT_WB, 1 / FS)
_ov_mask   = (_freqs_ov >= F_LO) & (_freqs_ov <= F_HI)
OV_FREQS   = _freqs_ov[_ov_mask] / 1000.0

_WIN_WB = np.blackman(N_FFT_WB)
_WIN_NB = np.blackman(N_FFT_NB)

DDR_BW_CHOICES  = ["5", "10", "25", "50"]      # kHz
DDR_MOD_CHOICES = ["--", "AM", "FM", "ASK"]
FFT_SIZE_CHOICES = ["64", "128", "256", "512", "1024"]

# DDR slot colours
DDR_COLORS = ["#FFA500", "#DA70D6"]   # orange, orchid

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#1e1e1e"
BG2      = "#252526"
BG3      = "#2d2d2d"
FG       = "#d4d4d4"
FG_DIM   = "#888888"
ACCENT   = "#0e639c"
ACCENT2  = "#1177bb"
ENTRY_BG = "#3c3c3c"
SEL_BG   = "#094771"

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


# ── Spectrum helpers ───────────────────────────────────────────────────────────
def _spectrum(ring: np.ndarray, window: np.ndarray, n_fft: int) -> np.ndarray:
    X = np.fft.rfft(ring * window)
    return 20 * np.log10(np.abs(X) / n_fft + 1e-12)


# ── DDC + Demodulation (batch, used for display/analysis) ─────────────────────
def ddc(ring: np.ndarray, fc_hz: float, bw_hz: float):
    """Digital down-convert ring to complex baseband. Returns (bb_complex64, fs_out)."""
    n     = len(ring)
    t     = np.arange(n) / FS
    bb    = ring.astype(np.complex128) * np.exp(-1j * 2 * np.pi * fc_hz * t)
    B     = np.fft.fft(bb)
    freqs = np.fft.fftfreq(n, 1 / FS)
    B[np.abs(freqs) > bw_hz / 2] = 0
    bb    = np.fft.ifft(B)
    dec   = max(1, int(FS / bw_hz))
    return bb[::dec].astype(np.complex64), float(FS / dec)


# ── DDReceiver (streaming model + audio stream) ────────────────────────────────
class DDReceiver:
    """
    Streaming DDC pipeline: every incoming UDP chunk is processed immediately
    so the audio buffer fills at exactly the stream drain rate.

    Phase continuity: _t_abs tracks the absolute sample index so the LO
    complex exponential is continuous across chunk boundaries.

    Decimation: integrate-and-dump (reshape + mean) with a leftover buffer
    so no samples are lost or double-counted across chunks.

    FM continuity: _prev_fm_angle carries the last unwrapped phase forward
    so there are no click artefacts at chunk boundaries.
    """

    def __init__(self):
        self.fc_hz    = None
        self.bw_hz    = 10_000.0
        self.mod_type = None
        self._playing = False
        self._dec     = 50                              # FS / bw_hz
        self._fs_out  = 10_000.0                       # bw_hz
        self._t_abs   = 0                              # absolute sample counter
        self._leftover = np.empty(0, dtype=np.complex128)  # fractional-dec remainder
        self._prev_fm_angle = 0.0                      # FM phase continuity
        self._buf     = collections.deque(maxlen=200_000)
        self._stream  = None
        self._rec_file: "IO | None" = None
        self._recording = False

    @property
    def playing(self):
        return self._playing

    @property
    def recording(self):
        return self._recording

    def configure(self, fc_hz: float, bw_hz: float):
        """Call whenever fc or BW changes. Resets streaming state; stops any recording."""
        was_playing = self._playing
        if was_playing:
            self.stop_playback()
        if self._recording:
            self.stop_recording()
        self.fc_hz   = fc_hz
        self.bw_hz   = bw_hz
        self._dec    = max(1, int(FS / bw_hz))
        self._fs_out = float(FS / self._dec)
        self._leftover      = np.empty(0, dtype=np.complex128)
        self._prev_fm_angle = 0.0
        if was_playing:
            self.start_playback()

    def push_chunk(self, chunk: np.ndarray):
        """Process one incoming UDP chunk through the streaming DDC pipeline."""
        self._t_abs += CHUNK
        if self.fc_hz is None:
            return

        # Phase-continuous complex mix to baseband
        t  = (np.arange(CHUNK, dtype=np.float64) + (self._t_abs - CHUNK)) / FS
        lo = np.exp(-1j * 2 * np.pi * self.fc_hz * t)
        bb = chunk.astype(np.float64) * lo

        # Integrate-and-dump decimation with leftover tracking
        dec  = self._dec
        work = np.concatenate([self._leftover, bb])
        n_full = len(work) // dec
        self._leftover = work[n_full * dec:]
        if n_full == 0:
            return
        bb_dec = work[:n_full * dec].reshape(n_full, dec).mean(axis=1).astype(np.complex64)

        # Write raw IQ to file
        if self._recording and self._rec_file:
            self._rec_file.write(bb_dec.tobytes())

        # Feed audio if playing
        if self.mod_type and self._playing:
            audio = self._demodulate(bb_dec)
            self._buf.extend(audio.tolist())

    def _demodulate(self, bb: np.ndarray) -> np.ndarray:
        if self.mod_type == "AM":
            out = np.abs(bb).astype(np.float32)
            out -= out.mean()                        # remove carrier DC per chunk
        elif self.mod_type == "FM":
            angles  = np.angle(bb).astype(np.float64)
            all_a   = np.concatenate([[self._prev_fm_angle], angles])
            self._prev_fm_angle = float(angles[-1])
            out     = np.diff(np.unwrap(all_a)).astype(np.float32)
            peak    = np.abs(out).max() + 1e-9
            out    /= peak
        elif self.mod_type == "ASK":
            env = np.abs(bb)
            out = ((env > env.mean()) * 2 - 1).astype(np.float32)
        else:
            out = np.zeros(len(bb), dtype=np.float32)
        return out

    def start_playback(self):
        self.stop_playback()
        if self.fc_hz is None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self._fs_out,
                channels=1,
                callback=self._sd_callback,
                dtype="float32",
                blocksize=256,
            )
            self._stream.start()
            self._playing = True
        except Exception as e:
            print(f"Audio stream error: {e}")

    def stop_playback(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._playing = False
        self._buf.clear()
        self._leftover      = np.empty(0, dtype=np.complex128)
        self._prev_fm_angle = 0.0

    def start_recording(self):
        """Open a new binary IQ file. Filename encodes fc, sample-rate, and timestamp."""
        self.stop_recording()
        if self.fc_hz is None:
            return
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fc_k = int(self.fc_hz / 1000)
        sr_k = int(self._fs_out / 1000)
        name = f"IQ_{fc_k}kHz_SR{sr_k}kHz_{ts}.bin"
        path = os.path.join(RECORDINGS_DIR, name)
        self._rec_file  = open(path, "wb")
        self._recording = True
        print(f"Recording IQ -> {name}")

    def stop_recording(self):
        if self._rec_file:
            self._rec_file.close()
            self._rec_file  = None
        self._recording = False

    def _sd_callback(self, outdata, frames, _time, _status):
        n = min(frames, len(self._buf))
        for i in range(n):
            outdata[i, 0] = self._buf.popleft()
        outdata[n:] = 0


# ── DDRPanel ───────────────────────────────────────────────────────────────────
class DDRPanel(tk.LabelFrame):
    def __init__(self, parent, idx: int, ddr: DDReceiver, canvas_ref,
                 linked: bool = False, **kw):
        color = DDR_COLORS[min(idx, len(DDR_COLORS) - 1)]
        label = " DDR " if linked else f" DDR-{idx+1} "
        super().__init__(parent, text=label, bg=BG2, fg=color,
                         font=("Consolas", 9, "bold"),
                         relief="flat", bd=1, highlightthickness=1,
                         highlightbackground=color, **kw)
        self._ddr        = ddr
        self._idx        = idx
        self._color      = color
        self._canvas_ref = canvas_ref
        self._linked     = linked     # True = NB mode; fc/bw come from channel
        self._build()

    def _build(self):
        pad = {"padx": 4, "pady": 3}

        if not self._linked:
            # Freq row
            freq_row = tk.Frame(self, bg=BG2)
            freq_row.pack(fill="x", **pad)
            tk.Label(freq_row, text="Freq (kHz)", bg=BG2, fg=FG,
                     font=("Consolas", 8)).pack(side="left")
            self._fc_var = tk.StringVar()
            self._fc_entry = tk.Entry(freq_row, textvariable=self._fc_var,
                                      bg=ENTRY_BG, fg=FG, insertbackground=FG,
                                      font=("Consolas", 9), width=8,
                                      relief="flat", bd=3)
            self._fc_entry.pack(side="left", padx=(4, 4))
            self._fc_entry.bind("<Return>", lambda _: self._apply())

            self._tune_btn = tk.Button(freq_row, text="Tune", font=("Consolas", 8),
                                       bg=BG3, fg=self._color,
                                       activebackground=self._color, activeforeground=BG,
                                       relief="flat", bd=0, padx=5, pady=1,
                                       cursor="hand2",
                                       command=self._start_tune)
            self._tune_btn.pack(side="left")

            # BW row
            bw_row = tk.Frame(self, bg=BG2)
            bw_row.pack(fill="x", **pad)
            tk.Label(bw_row, text="BW (kHz)  ", bg=BG2, fg=FG,
                     font=("Consolas", 8)).pack(side="left")
            self._bw_var = tk.StringVar(value="10")
            style = ttk.Style()
            style.configure("DDR.TCombobox", fieldbackground=ENTRY_BG,
                            background=BG3, foreground=FG,
                            selectbackground=SEL_BG, selectforeground="white",
                            arrowcolor=FG)
            bw_cb = ttk.Combobox(bw_row, textvariable=self._bw_var,
                                  values=DDR_BW_CHOICES,
                                  font=("Consolas", 8), width=6,
                                  state="readonly", style="DDR.TCombobox")
            bw_cb.pack(side="left")
            bw_cb.bind("<<ComboboxSelected>>", lambda _: self._apply())
        else:
            # Linked NB mode: show a read-only hint instead
            tk.Label(self, text="fc & BW locked to channel", bg=BG2, fg=FG_DIM,
                     font=("Consolas", 7), justify="left").pack(
                anchor="w", padx=6, pady=(4, 0))

        # Mod row (always shown)
        mod_row = tk.Frame(self, bg=BG2)
        mod_row.pack(fill="x", **pad)
        tk.Label(mod_row, text="Mod       ", bg=BG2, fg=FG,
                 font=("Consolas", 8)).pack(side="left")
        self._mod_var = tk.StringVar(value="--")
        mod_cb = ttk.Combobox(mod_row, textvariable=self._mod_var,
                               values=DDR_MOD_CHOICES,
                               font=("Consolas", 8), width=6,
                               state="readonly", style="DDR.TCombobox")
        mod_cb.pack(side="left")
        mod_cb.bind("<<ComboboxSelected>>", self._on_mod_change)

        # Play + Rec buttons on same row
        action_row = tk.Frame(self, bg=BG2)
        action_row.pack(fill="x", padx=4, pady=(2, 5))
        self._play_btn = tk.Button(action_row, text="Play", font=("Consolas", 8, "bold"),
                                   bg=BG3, fg=FG_DIM,
                                   activebackground=self._color, activeforeground=BG,
                                   relief="flat", bd=0, padx=10, pady=2,
                                   cursor="hand2", state="disabled",
                                   command=self._toggle_play)
        self._play_btn.pack(side="left", padx=(0, 6))
        self._rec_btn = tk.Button(action_row, text="Rec", font=("Consolas", 8, "bold"),
                                  bg=BG3, fg=FG_DIM,
                                  activebackground="#FF4444", activeforeground="white",
                                  relief="flat", bd=0, padx=8, pady=2,
                                  cursor="hand2", state="disabled",
                                  command=self._toggle_rec)
        self._rec_btn.pack(side="left")

    def set_canvas(self, canvas):
        self._canvas_ref = canvas

    def on_channel_applied(self):
        """Called by ChannelTab when NB channel Apply fires (linked mode only)."""
        if self._ddr.recording:
            self._ddr.stop_recording()
            self._rec_btn.config(text="Rec", bg=BG3, fg=FG_DIM)
        self._rec_btn.config(state="normal")
        mod = self._mod_var.get()
        self._play_btn.config(
            state="normal" if (mod != "--") else "disabled"
        )

    def _apply(self):
        try:
            fc_hz = float(self._fc_var.get()) * 1000
            if not (0 < fc_hz < FS / 2):
                return
        except ValueError:
            return
        bw_hz = float(self._bw_var.get()) * 1000
        # configure() stops any active recording (fc/bw changed = stale file)
        self._ddr.configure(fc_hz, bw_hz)
        if self._ddr.recording:
            self._rec_btn.config(text="Rec", bg=BG3, fg=FG_DIM)
        # Rec button enabled as soon as fc is configured (IQ capture is mod-agnostic)
        self._rec_btn.config(state="normal")
        if self._canvas_ref:
            self._canvas_ref.update_ddr_marker(self._idx, fc_hz, bw_hz)

    def _on_mod_change(self, _=None):
        mod = self._mod_var.get()
        self._ddr.mod_type = mod if mod != "--" else None
        enabled = (mod != "--") and (self._ddr.fc_hz is not None)
        self._play_btn.config(state="normal" if enabled else "disabled")
        if not enabled and self._ddr.playing:
            self._ddr.stop_playback()
            self._play_btn.config(text="Play", bg=BG3, fg=FG_DIM)

    def _toggle_play(self):
        if self._ddr.playing:
            self._ddr.stop_playback()
            self._play_btn.config(text="Play", bg=BG3, fg=FG_DIM)
        else:
            if self._ddr.fc_hz is None or not self._ddr.mod_type:
                return
            self._ddr.start_playback()
            self._play_btn.config(text="Stop", bg=self._color, fg=BG)

    def _toggle_rec(self):
        if self._ddr.recording:
            self._ddr.stop_recording()
            self._rec_btn.config(text="Rec", bg=BG3, fg=FG_DIM)
        else:
            if self._ddr.fc_hz is None:
                return
            self._ddr.start_recording()
            self._rec_btn.config(text="Stop", bg="#FF4444", fg="white")

    def _start_tune(self):
        if self._linked or not self._canvas_ref:
            return
        self._tune_btn.config(bg=self._color, fg=BG)
        self._canvas_ref.set_tune_listener(self._on_tune_click)

    def _on_tune_click(self, fc_hz: float):
        self._tune_btn.config(bg=BG3, fg=self._color)
        self._fc_var.set(f"{fc_hz/1000:.2f}")
        self._apply()
        mod = self._mod_var.get()
        self._play_btn.config(
            state="normal" if (mod != "--") else "disabled"
        )

    def destroy(self):
        self._ddr.stop_playback()
        self._ddr.stop_recording()
        super().destroy()


# ── ChannelCanvas ──────────────────────────────────────────────────────────────
class ChannelCanvas(tk.Frame):
    _DEFAULT_N_FFT = {"WB": 512, "NB": 1024}

    def __init__(self, parent, ch_type: str, color: str):
        super().__init__(parent, bg=BG2)
        self._ch_type  = ch_type
        self._color    = color
        self._mask     = None
        self._freqs    = None
        self._tune_cb  = None
        self._n_fft    = self._DEFAULT_N_FFT.get(ch_type, 512)
        self._window   = np.blackman(self._n_fft)

        # DDR marker artists: list of dicts, one per DDR slot
        self._ddr_artists = []   # filled by add_ddr_markers()

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
        self._ax     = ax

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._fig_canvas = canvas

        fig.canvas.mpl_connect("button_press_event", self._on_mpl_click)

    def add_ddr_markers(self, count: int):
        """Called once at ChannelTab init to pre-create hidden DDR marker artists."""
        for i in range(count):
            color = DDR_COLORS[i]
            vline = self._ax.axvline(0, color=color, linewidth=1.2, visible=False)
            span  = self._ax.axvspan(0, 1, alpha=0.12, color=color, visible=False)
            label = self._ax.text(0, 8, f"DDR-{i+1}", color=color,
                                  fontsize=7, visible=False,
                                  fontfamily="Consolas")
            self._ddr_artists.append({"vline": vline, "span": span, "label": label})

    def update_ddr_marker(self, idx: int, fc_hz: float, bw_hz: float):
        if idx >= len(self._ddr_artists):
            return
        m   = self._ddr_artists[idx]
        fc_k = fc_hz / 1000
        lo_k = (fc_hz - bw_hz / 2) / 1000
        hi_k = (fc_hz + bw_hz / 2) / 1000
        m["vline"].set_xdata([fc_k, fc_k])
        m["vline"].set_visible(True)
        # axvspan has no simple setter; hide old and note span extent via xy
        m["span"].set_xy([[lo_k, 0], [lo_k, 1], [hi_k, 1], [hi_k, 0], [lo_k, 0]])
        m["span"].set_visible(True)
        m["label"].set_x(fc_k)
        m["label"].set_visible(True)
        self._fig_canvas.draw_idle()

    def clear_ddr_marker(self, idx: int):
        if idx >= len(self._ddr_artists):
            return
        m = self._ddr_artists[idx]
        m["vline"].set_visible(False)
        m["span"].set_visible(False)
        m["label"].set_visible(False)
        self._fig_canvas.draw_idle()

    def set_tune_listener(self, callback):
        self._tune_cb = callback
        self._fig_canvas.get_tk_widget().config(cursor="crosshair")

    def _on_mpl_click(self, event):
        if event.inaxes and self._tune_cb:
            cb = self._tune_cb
            self._tune_cb = None
            self._fig_canvas.get_tk_widget().config(cursor="")
            cb(event.xdata * 1000)

    def setup(self, fc_hz: float, bw_hz: float, n_fft: int = None):
        if n_fft is not None:
            self._n_fft  = n_fft
            self._window = np.blackman(n_fft)
        freqs = np.fft.rfftfreq(self._n_fft, 1 / FS)
        lo    = fc_hz - bw_hz / 2
        hi    = fc_hz + bw_hz / 2
        self._mask  = (freqs >= lo) & (freqs <= hi)
        self._freqs = freqs[self._mask] / 1000.0

        fc_k = fc_hz / 1000
        self._ax.set_xlim(lo / 1000, hi / 1000)
        res = FS / self._n_fft
        self._ax.set_title(
            f"{self._ch_type}  fc={fc_k:.1f} kHz  BW={bw_hz/1000:.1f} kHz"
            f"  N={self._n_fft}  res~{res:.0f} Hz/bin",
            color=self._color, fontsize=9,
        )
        self._line.set_data([], [])
        self._fig_canvas.draw_idle()

    def refresh(self, ring_wb: np.ndarray, ring_nb: np.ndarray):
        if self._mask is None:
            return
        ring = ring_nb if self._ch_type == "NB" else ring_wb
        seg  = ring[-self._n_fft:]
        mag  = _spectrum(seg, self._window, self._n_fft)
        self._line.set_data(self._freqs, mag[self._mask])
        self._fig_canvas.draw_idle()


# ── ParamPanel ─────────────────────────────────────────────────────────────────
class ParamPanel(tk.Frame):
    _BW_LIMITS = {"WB": (20_000, 200_000), "NB": (1_000, 20_000)}

    def __init__(self, parent, ch_type: str, on_apply):
        super().__init__(parent, bg=BG2, padx=14, pady=10, width=230)
        self.pack_propagate(False)
        self._ch_type  = ch_type
        self._on_apply = on_apply

        lo, hi  = self._BW_LIMITS[ch_type]
        bw_hint = f"{lo//1000} - {hi//1000} kHz"
        bw_def  = "60" if ch_type == "WB" else "10"
        hdr_col = "#00BFFF" if ch_type == "WB" else "#00FF9F"

        tk.Label(self, text=f"{ch_type} Receiver", bg=BG2, fg=hdr_col,
                 font=("Consolas", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        tk.Label(self, text="Center Freq (kHz)", bg=BG2, fg=FG,
                 font=("Consolas", 9)).grid(row=1, column=0, sticky="w", pady=3)
        self._fc_var = tk.StringVar(value="100")
        tk.Entry(self, textvariable=self._fc_var, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=("Consolas", 9),
                 width=10, relief="flat", bd=4).grid(
            row=1, column=1, sticky="w", padx=(8, 0))

        tk.Label(self, text=f"Bandwidth (kHz)\n({bw_hint})", bg=BG2, fg=FG,
                 font=("Consolas", 9), justify="left").grid(
            row=2, column=0, sticky="w", pady=3)
        self._bw_var = tk.StringVar(value=bw_def)
        tk.Entry(self, textvariable=self._bw_var, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=("Consolas", 9),
                 width=10, relief="flat", bd=4).grid(
            row=2, column=1, sticky="w", padx=(8, 0))

        fft_default = "1024" if ch_type == "NB" else "512"
        tk.Label(self, text="FFT Size", bg=BG2, fg=FG,
                 font=("Consolas", 9)).grid(row=3, column=0, sticky="w", pady=3)
        self._fft_var = tk.StringVar(value=fft_default)
        _fft_style = ttk.Style()
        _fft_style.configure("Param.TCombobox", fieldbackground=ENTRY_BG,
                             background=BG3, foreground=FG,
                             selectbackground=SEL_BG, selectforeground="white",
                             arrowcolor=FG)
        ttk.Combobox(self, textvariable=self._fft_var,
                     values=FFT_SIZE_CHOICES,
                     font=("Consolas", 9), width=8,
                     state="readonly", style="Param.TCombobox").grid(
            row=3, column=1, sticky="w", padx=(8, 0), pady=3)

        tk.Frame(self, bg=BG3, height=1).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=8)

        tk.Button(self, text="Apply", command=self._apply,
                  bg=ACCENT, fg=FG, activebackground=ACCENT2, activeforeground="white",
                  relief="flat", font=("Consolas", 9), padx=10, pady=4,
                  cursor="hand2", bd=0).grid(row=5, column=0, columnspan=2, sticky="w")

        self._info = tk.Label(self, text="", bg=BG2, fg=FG_DIM,
                              font=("Consolas", 8), wraplength=200, justify="left")
        self._info.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _apply(self):
        nyq = FS / 2
        try:
            fc_hz = float(self._fc_var.get()) * 1000
            if not (0 < fc_hz < nyq):
                messagebox.showerror("Input Error",
                                     f"Center freq must be 0 - {nyq/1000:.0f} kHz.")
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
                    f"Bandwidth must be {lo//1000}-{hi//1000} kHz for {self._ch_type}.",
                )
                return
        except ValueError:
            messagebox.showerror("Input Error", "Bandwidth must be a number.")
            return
        n_fft = int(self._fft_var.get())
        res   = FS / n_fft
        self._info.config(
            text=f"fc = {fc_hz/1000:.1f} kHz\nBW = {bw_hz/1000:.1f} kHz"
                 f"\nN = {n_fft}  res ~ {res:.0f} Hz/bin"
        )
        self._on_apply(fc_hz, bw_hz, n_fft)


# ── ChannelTab ─────────────────────────────────────────────────────────────────
class ChannelTab(tk.Frame):
    def __init__(self, parent, name: str, ch_type: str, color: str, state: dict):
        super().__init__(parent, bg=BG)
        self.name    = name
        self.ch_type = ch_type
        self._state  = state
        self._active = False
        self._ddrs:       list[DDReceiver] = []
        self._ddr_panels: list[DDRPanel]   = []

        # Left column: scrollable frame for params + DDR panels
        left_outer = tk.Frame(self, bg=BG2, width=240)
        left_outer.pack(side="left", fill="y")
        left_outer.pack_propagate(False)

        canvas_scroll = tk.Canvas(left_outer, bg=BG2, highlightthickness=0)
        scrollbar     = tk.Scrollbar(left_outer, orient="vertical",
                                     command=canvas_scroll.yview,
                                     bg=BG3, troughcolor=BG2, width=8)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)

        self._left = tk.Frame(canvas_scroll, bg=BG2)
        win_id = canvas_scroll.create_window((0, 0), window=self._left,
                                             anchor="nw")
        self._left.bind("<Configure>",
                        lambda e: canvas_scroll.configure(
                            scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.bind("<Configure>",
                           lambda e: canvas_scroll.itemconfig(
                               win_id, width=e.width))

        # Right: spectrum canvas
        sep = tk.Frame(self, bg=BG3, width=1)
        sep.pack(side="left", fill="y")

        self._ch_canvas = ChannelCanvas(self, ch_type=ch_type, color=color)
        self._ch_canvas.pack(side="left", fill="both", expand=True)

        # Param panel
        params = ParamPanel(self._left, ch_type=ch_type, on_apply=self._on_apply)
        params.pack(fill="x", pady=(0, 0))

        # DDR panels
        tk.Frame(self._left, bg=BG3, height=1).pack(fill="x", padx=6, pady=6)
        if ch_type == "WB":
            self._ch_canvas.add_ddr_markers(2)
            for i in range(2):
                ddr   = DDReceiver()
                panel = DDRPanel(self._left, idx=i, ddr=ddr,
                                 canvas_ref=self._ch_canvas,
                                 padx=6, pady=6)
                panel.pack(fill="x", padx=6, pady=(0, 6))
                self._ddrs.append(ddr)
                self._ddr_panels.append(panel)
        elif ch_type == "NB":
            ddr   = DDReceiver()
            panel = DDRPanel(self._left, idx=0, ddr=ddr,
                             canvas_ref=None, linked=True,
                             padx=6, pady=6)
            panel.pack(fill="x", padx=6, pady=(0, 6))
            self._ddrs.append(ddr)
            self._ddr_panels.append(panel)

    def _on_apply(self, fc_hz: float, bw_hz: float, n_fft: int):
        self._ch_canvas.setup(fc_hz, bw_hz, n_fft)
        self._active = True
        if self.ch_type == "NB" and self._ddrs:
            self._ddrs[0].configure(fc_hz, bw_hz)
            self._ddr_panels[0].on_channel_applied()

    def push_chunk(self, chunk: np.ndarray):
        """Forward each raw UDP chunk to all DDRs for streaming processing."""
        for ddr in self._ddrs:
            ddr.push_chunk(chunk)

    def update(self):
        """Refresh the spectrum display (DDR audio is handled per-chunk in push_chunk)."""
        if not self._active:
            return
        self._ch_canvas.refresh(self._state["ring_wb"], self._state["ring_nb"])

    def destroy(self):
        for ddr in self._ddrs:
            ddr.stop_playback()
            ddr.stop_recording()
        super().destroy()


# ── OverviewTab ────────────────────────────────────────────────────────────────
class OverviewTab(tk.Frame):
    def __init__(self, parent, app, state: dict):
        super().__init__(parent, bg=BG)
        self._app        = app
        self._state      = state
        self._btn_active = {n: False for n in CHANNEL_DEFS}
        self._btns: dict[str, tk.Button] = {}
        self._got_signal = False
        self._build()

    def _build(self):
        fig = plt.Figure(figsize=(14, 7), facecolor="#0d0d0d")
        fig.suptitle("Signal Capture  -  Live Spectrum",
                     fontsize=13, color="white", y=0.99)

        gs    = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[7, 3],
                                  hspace=0.38, left=0.07, right=0.97,
                                  top=0.94, bottom=0.08)
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
        ax_td.set_title("Composite waveform (rolling 10 ms)",
                        color="#888888", fontsize=8)

        t_ms = np.linspace(0, TD_SECS * 1000, TD_SAMPLES)
        self._spec_line, = ax_sp.plot([], [], color="#00BFFF", linewidth=0.9)
        self._td_line,   = ax_td.plot(t_ms, np.zeros(TD_SAMPLES),
                                       color="#00FF9F", linewidth=0.5)
        self._ax_sp = ax_sp

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas = canvas

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
        self.geometry("1280x760")

        _ensure_recordings_dir()

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
            # Stream each raw chunk through all active channel DDRs immediately
            for ch in self._channels.values():
                ch.push_chunk(chunk)
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
