"""
AM Modulation RF Simulation
Multiple AM signals with different carrier frequencies around 150 MHz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Simulation Parameters ───────────────────────────────────────────────────
FS = 2e9          # Sample rate: 2 GHz (must be > 2x highest carrier)
DURATION = 1e-4   # 100 µs capture window
N = int(FS * DURATION)

t = np.linspace(0, DURATION, N, endpoint=False)

# ─── AM Signal Definitions ───────────────────────────────────────────────────
# Each entry: (carrier_freq_MHz, modulation_freq_kHz, mod_index, label)
SIGNALS = [
    (147.5e6,  1.0e3, 0.5,  "CH1 - 147.5 MHz"),
    (149.0e6,  2.5e3, 0.7,  "CH2 - 149.0 MHz"),
    (150.5e6,  5.0e3, 0.4,  "CH3 - 150.5 MHz"),
    (152.0e6, 10.0e3, 0.8,  "CH4 - 152.0 MHz"),
    (153.5e6,  3.0e3, 0.6,  "CH5 - 153.5 MHz"),
]

COLORS = ["#00BFFF", "#FF6B6B", "#00FF9F", "#FFD700", "#BF5FFF"]

# ─── Generate AM Signals ──────────────────────────────────────────────────────
def am_signal(t, fc, fm, m):
    """
    AM signal: s(t) = [1 + m·cos(2π·fm·t)] · cos(2π·fc·t)
    fc  = carrier frequency
    fm  = modulating (audio) frequency
    m   = modulation index (0 < m ≤ 1)
    """
    carrier   = np.cos(2 * np.pi * fc * t)
    modulator = 1 + m * np.cos(2 * np.pi * fm * t)
    return modulator * carrier

composite = np.zeros(N)
signals   = []

for fc, fm, m, label in SIGNALS:
    s = am_signal(t, fc, fm, m)
    signals.append(s)
    composite += s

# ─── FFT ─────────────────────────────────────────────────────────────────────
def compute_fft(x, fs):
    win      = np.blackman(len(x))           # Blackman window → lower sidelobes
    X        = np.fft.rfft(x * win)
    freqs    = np.fft.rfftfreq(len(x), 1/fs)
    mag_db   = 20 * np.log10(np.abs(X) / len(x) + 1e-12)
    return freqs, mag_db

freqs, composite_db = compute_fft(composite, FS)

# Frequency range to display: 145 – 156 MHz
f_lo, f_hi = 145e6, 156e6
mask = (freqs >= f_lo) & (freqs <= f_hi)
f_view   = freqs[mask]
db_view  = composite_db[mask]

# Per-channel FFTs for the individual-channel subplot
channel_ffts = []
for s in signals:
    _, db = compute_fft(s, FS)
    channel_ffts.append(db[mask])

# ─── Plot ─────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
fig.suptitle("AM RF Simulation  ·  Carriers around 150 MHz",
             fontsize=15, color="white", y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.55, wspace=0.35,
                       left=0.07, right=0.97, top=0.93, bottom=0.07)

# 1) Composite FFT spectrum (spans full width, top row)
ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(f_view / 1e6, db_view, color="#00BFFF", linewidth=1.0, label="Composite")
ax_main.set_title("Composite FFT Spectrum", color="white", fontsize=11)
ax_main.set_xlabel("Frequency (MHz)", color="gray")
ax_main.set_ylabel("Power (dBFS)", color="gray")
ax_main.tick_params(colors="gray")
ax_main.set_facecolor("#111111")
ax_main.grid(True, color="#2a2a2a", linewidth=0.6)
ax_main.set_xlim(f_lo / 1e6, f_hi / 1e6)
ax_main.set_ylim(-90, 10)

# Annotate carrier peaks
for (fc, fm, m, label), color in zip(SIGNALS, COLORS):
    ax_main.axvline(fc / 1e6, color=color, linestyle="--", alpha=0.5, linewidth=0.8)
    ax_main.text(fc / 1e6, 5, f"{fc/1e6:.1f}", color=color,
                 fontsize=7.5, ha="center", va="bottom", rotation=90)

# 2) Individual channel spectra (rows 1–2, two columns)
axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

# Use only 4 slots; CH5 overlays on main (add a 5th subplot if needed)
for i in range(min(4, len(SIGNALS))):
    ax = axes[i]
    fc, fm, m, label = SIGNALS[i]
    ax.plot(f_view / 1e6, channel_ffts[i], color=COLORS[i], linewidth=0.9)
    ax.set_title(f"{label}  (m={m}, fm={fm/1e3:.1f} kHz)",
                 color=COLORS[i], fontsize=9)
    ax.set_xlabel("Frequency (MHz)", color="gray", fontsize=8)
    ax.set_ylabel("dBFS", color="gray", fontsize=8)
    ax.tick_params(colors="gray", labelsize=7)
    ax.set_facecolor("#111111")
    ax.grid(True, color="#2a2a2a", linewidth=0.5)
    ax.set_xlim(f_lo / 1e6, f_hi / 1e6)
    ax.set_ylim(-90, 10)
    # Mark carrier + sidebands
    ax.axvline(fc / 1e6, color=COLORS[i], linestyle="--", alpha=0.7, linewidth=0.8)
    ax.axvline((fc + fm) / 1e6, color="white", linestyle=":", alpha=0.4, linewidth=0.7)
    ax.axvline((fc - fm) / 1e6, color="white", linestyle=":", alpha=0.4, linewidth=0.7)

# 5th channel sits in ax_main legend area — add text annotation instead
ax_main.plot(f_view / 1e6, channel_ffts[4], color=COLORS[4],
             linewidth=0.7, alpha=0.6, linestyle="-", label=SIGNALS[4][3])
ax_main.legend(loc="upper right", fontsize=8, facecolor="#1a1a1a",
               edgecolor="#333", labelcolor="white")

# ─── Time-domain inset (small, inside composite plot) ────────────────────────
ax_td = ax_main.inset_axes([0.01, 0.05, 0.18, 0.45])
t_us   = t * 1e6
n_show = int(FS * 10e-6)   # show 10 µs
ax_td.plot(t_us[:n_show], composite[:n_show], color="#00BFFF", linewidth=0.4)
ax_td.set_title("Time (10 µs)", color="gray", fontsize=6)
ax_td.tick_params(colors="gray", labelsize=5)
ax_td.set_facecolor("#0a0a0a")
ax_td.grid(True, color="#222", linewidth=0.4)
ax_td.set_xlabel("µs", color="gray", fontsize=5)

plt.savefig("C:/pytest/am_spectrum.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Plot saved -> C:/pytest/am_spectrum.png")
