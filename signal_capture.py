"""
AM Signal Capture & Spectrum Visualizer
Receives UDP chunks from am_simulation.py, maintains a ring buffer,
and displays a live FFT spectrum + rolling time-domain waveform.
"""

import socket
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

from config import (FS, N_FFT, CHUNK, F_LO, F_HI,
                    SIGNALS, COLORS, UDP_HOST, UDP_PORT)

PACKET_BYTES = CHUNK * 4       # float32
TD_SECS      = 0.010           # time-domain strip length: 10 ms
TD_SAMPLES   = int(FS * TD_SECS)
INTERVAL_MS  = 33              # animation frame interval (~30 fps)

# Precompute frequency axis for the display window
_freqs_all = np.fft.rfftfreq(N_FFT, 1 / FS)
_mask      = (_freqs_all >= F_LO) & (_freqs_all <= F_HI)
FREQS_KHZ  = _freqs_all[_mask] / 1000.0


# ─── UDP receiver thread ──────────────────────────────────────────────────────
def receiver_thread(pkt_queue: queue.Queue, stop_event: threading.Event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.settimeout(0.5)
    print(f"Listening on {UDP_HOST}:{UDP_PORT} ...")

    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(PACKET_BYTES)
            if len(data) == PACKET_BYTES:
                chunk = np.frombuffer(data, dtype=np.float32)
                pkt_queue.put(chunk)
        except socket.timeout:
            continue

    sock.close()


# ─── Spectrum computation ─────────────────────────────────────────────────────
_window = np.blackman(N_FFT)

def compute_spectrum(ring: np.ndarray) -> np.ndarray:
    X      = np.fft.rfft(ring * _window)
    mag_db = 20 * np.log10(np.abs(X) / N_FFT + 1e-12)
    return mag_db[_mask]


# ─── Figure setup ─────────────────────────────────────────────────────────────
def build_figure():
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 7), facecolor="#0d0d0d")
    fig.suptitle("Signal Capture  ·  Live Spectrum", fontsize=13, color="white", y=0.99)

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[7, 3],
                           hspace=0.38, left=0.07, right=0.97, top=0.94, bottom=0.08)

    # ── Spectrum panel ──
    ax_spec = fig.add_subplot(gs[0])
    ax_spec.set_facecolor("#111111")
    ax_spec.set_xlim(F_LO / 1000, F_HI / 1000)
    ax_spec.set_ylim(-90, 10)
    ax_spec.set_xlabel("Frequency (kHz)", color="gray")
    ax_spec.set_ylabel("Power (dBFS)", color="gray")
    ax_spec.tick_params(colors="gray")
    ax_spec.grid(True, color="#2a2a2a", linewidth=0.6)
    ax_spec.set_title("Waiting for signal...", color="#555555", fontsize=9)

    for (fc, fm, m, label), color in zip(SIGNALS, COLORS):
        ax_spec.axvline(fc / 1000, color=color, linestyle="--", alpha=0.5, linewidth=0.8)
        ax_spec.text(fc / 1000, 5, f"{fc/1000:.0f} kHz\n{label}",
                     color=color, fontsize=7.5, ha="center", va="bottom")

    spec_line, = ax_spec.plot([], [], color="#00BFFF", linewidth=0.9, animated=True)

    # ── Time-domain panel ──
    ax_td = fig.add_subplot(gs[1])
    ax_td.set_facecolor("#0a0a0a")
    ax_td.set_xlim(0, TD_SECS * 1000)
    ax_td.set_ylim(-6, 6)
    ax_td.set_xlabel("Time (ms)", color="gray", fontsize=9)
    ax_td.set_ylabel("Amplitude", color="gray", fontsize=9)
    ax_td.tick_params(colors="gray", labelsize=7)
    ax_td.grid(True, color="#222222", linewidth=0.4)
    ax_td.set_title("Composite waveform (rolling 10 ms)", color="#888888", fontsize=8)

    t_ms = np.linspace(0, TD_SECS * 1000, TD_SAMPLES)
    td_line, = ax_td.plot(t_ms, np.zeros(TD_SAMPLES),
                          color="#00FF9F", linewidth=0.5, animated=True)

    return fig, ax_spec, ax_td, spec_line, td_line


# ─── Animation ────────────────────────────────────────────────────────────────
def make_init(spec_line, td_line):
    def init():
        spec_line.set_data([], [])
        td_line.set_ydata(np.zeros(TD_SAMPLES))
        return spec_line, td_line
    return init


def make_update(spec_line, td_line, ax_spec, state, pkt_queue):
    got_signal = [False]

    def update(_frame):
        # drain all pending packets into the ring buffer
        updated = False
        while not pkt_queue.empty():
            try:
                chunk = pkt_queue.get_nowait()
            except queue.Empty:
                break

            state["ring"][:-CHUNK] = state["ring"][CHUNK:]
            state["ring"][-CHUNK:] = chunk

            state["td_buffer"][:-CHUNK] = state["td_buffer"][CHUNK:]
            state["td_buffer"][-CHUNK:] = chunk

            updated = True

        if updated:
            if not got_signal[0]:
                ax_spec.set_title("", color="gray", fontsize=9)
                got_signal[0] = True
            spec_line.set_data(FREQS_KHZ, compute_spectrum(state["ring"]))
            td_line.set_ydata(state["td_buffer"])

        return spec_line, td_line

    return update


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pkt_queue  = queue.Queue(maxsize=200)
    stop_event = threading.Event()

    rx = threading.Thread(target=receiver_thread,
                          args=(pkt_queue, stop_event), daemon=True)
    rx.start()

    fig, ax_spec, ax_td, spec_line, td_line = build_figure()

    state = {
        "ring":      np.zeros(N_FFT),
        "td_buffer": np.zeros(TD_SAMPLES),
    }

    anim = FuncAnimation(
        fig,
        make_update(spec_line, td_line, ax_spec, state, pkt_queue),
        init_func=make_init(spec_line, td_line),
        interval=INTERVAL_MS,
        blit=True,
        cache_frame_data=False,
    )

    try:
        plt.show()
    finally:
        stop_event.set()
