"""
AM Modulation RF Simulation — Live Streaming Spectrum Analyzer
5 AM channels, continuously updated FFT display + rolling time-domain strip
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ─── Parameters ──────────────────────────────────────────────────────────────
FS        = 500_000      # Sample rate: 500 kHz
N_FFT     = 4096         # FFT size → ~122 Hz/bin resolution
CHUNK     = 512          # New samples per animation frame (87.5% overlap)
INTERVAL  = 33           # ms between frames (~30 fps)
F_LO      = 40_000       # Display window low edge (Hz)
F_HI      = 240_000      # Display window high edge (Hz)
NOISE_STD = 0.015        # AWGN amplitude → ~−75 dBFS noise floor after FFT
TD_SECS   = 0.010        # Time-domain strip length: 10 ms

# carrier_hz, mod_hz, mod_index, label
SIGNALS = [
    ( 60_000,  1_000, 0.50, "CH1"),
    ( 90_000,  2_500, 0.70, "CH2"),
    (130_000,  5_000, 0.40, "CH3"),
    (170_000, 10_000, 0.80, "CH4"),
    (210_000,  3_000, 0.60, "CH5"),
]
COLORS = ["#00BFFF", "#FF6B6B", "#00FF9F", "#FFD700", "#BF5FFF"]

# Precompute frequency-bin mask for the display window
_freqs_all = np.fft.rfftfreq(N_FFT, 1 / FS)
_mask      = (_freqs_all >= F_LO) & (_freqs_all <= F_HI)
FREQS_KHZ  = _freqs_all[_mask] / 1000.0   # display in kHz, computed once

TD_SAMPLES = int(FS * TD_SECS)


# ─── Signal generation ────────────────────────────────────────────────────────
def generate_chunk(t_offset: float) -> np.ndarray:
    """Generate CHUNK samples of composite AM + AWGN, phase-continuous across calls."""
    t = np.arange(CHUNK) / FS + t_offset
    composite = np.zeros(CHUNK)
    for fc, fm, m, _ in SIGNALS:
        composite += (1 + m * np.cos(2 * np.pi * fm * t)) * np.cos(2 * np.pi * fc * t)
    composite += np.random.randn(CHUNK) * NOISE_STD
    return composite


# ─── Spectrum computation ─────────────────────────────────────────────────────
def compute_spectrum(ring: np.ndarray):
    """Blackman-windowed rfft on the ring buffer → dBFS, sliced to display window."""
    win    = np.blackman(N_FFT)
    X      = np.fft.rfft(ring * win)
    mag_db = 20 * np.log10(np.abs(X) / N_FFT + 1e-12)
    return mag_db[_mask]


# ─── Figure setup ─────────────────────────────────────────────────────────────
def build_figure():
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 7), facecolor="#0d0d0d")
    fig.suptitle("AM Spectrum Analyzer  ·  Live",
                 fontsize=13, color="white", y=0.99)

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[7, 3],
                           hspace=0.35, left=0.07, right=0.97, top=0.94, bottom=0.08)

    # ── Spectrum panel ──
    ax_spec = fig.add_subplot(gs[0])
    ax_spec.set_facecolor("#111111")
    ax_spec.set_xlim(F_LO / 1000, F_HI / 1000)
    ax_spec.set_ylim(-90, 10)
    ax_spec.set_xlabel("Frequency (kHz)", color="gray")
    ax_spec.set_ylabel("Power (dBFS)", color="gray")
    ax_spec.tick_params(colors="gray")
    ax_spec.grid(True, color="#2a2a2a", linewidth=0.6)

    for (fc, fm, m, label), color in zip(SIGNALS, COLORS):
        ax_spec.axvline(fc / 1000, color=color, linestyle="--", alpha=0.5, linewidth=0.8)
        ax_spec.text(fc / 1000, 5, f"{fc/1000:.0f} kHz\n{label}",
                     color=color, fontsize=7.5, ha="center", va="bottom")

    spec_line, = ax_spec.plot([], [], color="#00BFFF", linewidth=0.9, animated=True)

    # ── Time-domain panel ──
    ax_td = fig.add_subplot(gs[1])
    ax_td.set_facecolor("#0a0a0a")
    t_ms = np.linspace(0, TD_SECS * 1000, TD_SAMPLES)
    ax_td.set_xlim(0, TD_SECS * 1000)
    ax_td.set_ylim(-6, 6)
    ax_td.set_xlabel("Time (ms)", color="gray", fontsize=9)
    ax_td.set_ylabel("Amplitude", color="gray", fontsize=9)
    ax_td.tick_params(colors="gray", labelsize=7)
    ax_td.grid(True, color="#222222", linewidth=0.4)
    ax_td.set_title("Composite waveform (rolling 10 ms)", color="#888888", fontsize=8)

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


def make_update(spec_line, td_line, state):
    def update(_frame):
        chunk = generate_chunk(state["t_offset"])
        state["t_offset"] += CHUNK / FS

        # Roll ring buffer
        ring = state["ring"]
        ring[:-CHUNK] = ring[CHUNK:]
        ring[-CHUNK:]  = chunk

        # Roll time-domain buffer
        td = state["td_buffer"]
        td[:-CHUNK] = td[CHUNK:]
        td[-CHUNK:]  = chunk

        spec_line.set_data(FREQS_KHZ, compute_spectrum(ring))
        td_line.set_ydata(td)
        return spec_line, td_line
    return update


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig, ax_spec, ax_td, spec_line, td_line = build_figure()

    state = {
        "ring":      np.zeros(N_FFT),
        "t_offset":  0.0,
        "td_buffer": np.zeros(TD_SAMPLES),
    }

    anim = FuncAnimation(
        fig,
        make_update(spec_line, td_line, state),
        init_func=make_init(spec_line, td_line),
        interval=INTERVAL,
        blit=True,
        cache_frame_data=False,
    )

    plt.show()
