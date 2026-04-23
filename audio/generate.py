"""
Generates synthetic radio communication WAV files for use as modulating signals.
Run this script directly to (re)generate all files in this folder.
"""

import os
import wave
import numpy as np

FS = 8000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_wav(filename: str, samples: np.ndarray):
    s16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    path = os.path.join(OUT_DIR, filename)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(s16.tobytes())
    print(f"  {filename}: {len(samples)/FS:.2f}s  ({len(samples)} samples)")


def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * FS)) / FS


def _sine(freq: float, dur: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * _t(dur))


def _silence(dur: float) -> np.ndarray:
    return np.zeros(int(dur * FS))


def _fade(s: np.ndarray, ms: float = 8.0) -> np.ndarray:
    n = int(ms * FS / 1000)
    n = min(n, len(s) // 2)
    s = s.copy()
    s[:n]  *= np.linspace(0, 1, n)
    s[-n:] *= np.linspace(1, 0, n)
    return s


def _normalise(s: np.ndarray, peak: float = 0.92) -> np.ndarray:
    m = np.abs(s).max()
    return s * (peak / m) if m > 0 else s


# ── 1. Morse SOS  (... --- ...) ───────────────────────────────────────────────
def make_morse_sos() -> np.ndarray:
    freq   = 700.0
    dot    = 0.075
    dash   = 0.225
    eg     = 0.075   # element gap
    lg     = 0.225   # letter gap
    wg     = 0.60    # word gap

    def dit():  return _fade(_sine(freq, dot))
    def dah():  return _fade(_sine(freq, dash))

    def letter(elements):
        parts = []
        for i, e in enumerate(elements):
            parts.append(e)
            if i < len(elements) - 1:
                parts.append(_silence(eg))
        return np.concatenate(parts)

    S = letter([dit(), dit(), dit()])
    O = letter([dah(), dah(), dah()])

    sos = np.concatenate([S, _silence(lg), O, _silence(lg), S, _silence(wg)])
    # repeat once with a pause
    return _normalise(np.concatenate([sos, sos]))


# ── 2. Voice – male  (130 Hz fundamental, speech-rhythm AM) ──────────────────
def make_voice_male() -> np.ndarray:
    dur = 3.2
    t   = _t(dur)
    f0  = 130.0

    sig = np.zeros_like(t)
    for k in range(1, 14):
        amp = (1.0 / k) * np.exp(-0.25 * (k - 1))
        sig += amp * np.sin(2 * np.pi * k * f0 * t)

    # syllable rate ≈ 4 Hz, word rate ≈ 1.5 Hz
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 1.4 * t - 0.3)) \
        * (0.5 + 0.5 * np.sin(2 * np.pi * 4.1 * t + 0.7))
    env = np.clip(env, 0, 1) ** 0.6

    # breathy noise layer (voiced fricatives)
    noise = np.random.RandomState(1).randn(len(t)) * 0.08
    sig   = sig * env + noise * env

    sig = _fade(sig, ms=40)
    return _normalise(sig)


# ── 3. Voice – female  (220 Hz fundamental) ──────────────────────────────────
def make_voice_female() -> np.ndarray:
    dur = 3.0
    t   = _t(dur)
    f0  = 220.0

    sig = np.zeros_like(t)
    for k in range(1, 10):
        amp = (1.0 / k ** 0.75) * np.exp(-0.18 * (k - 1))
        sig += amp * np.sin(2 * np.pi * k * f0 * t)

    env = (0.4 + 0.6 * np.sin(2 * np.pi * 1.8 * t + 1.0)) \
        * (0.3 + 0.7 * np.sin(2 * np.pi * 5.2 * t + 0.2))
    env = np.clip(env, 0, 1) ** 0.5

    noise = np.random.RandomState(2).randn(len(t)) * 0.06
    sig   = sig * env + noise * env

    sig = _fade(sig, ms=40)
    return _normalise(sig)


# ── 4. AFSK data burst  (1200 / 2200 Hz, 300 baud, Bell 103-like) ────────────
def make_data_afsk() -> np.ndarray:
    baud   = 300
    f_mark = 1200.0   # '1'
    f_spc  = 2200.0   # '0'
    bits_per_sym = 1
    sym_dur = 1.0 / baud

    rng  = np.random.RandomState(42)
    bits = rng.randint(0, 2, int(2.5 * baud))  # ~2.5 s of data

    parts = []
    phase = 0.0
    for bit in bits:
        freq = f_mark if bit else f_spc
        n    = int(sym_dur * FS)
        t    = np.arange(n) / FS
        chunk = np.sin(2 * np.pi * freq * t + phase)
        # maintain phase continuity across symbols
        phase = (phase + 2 * np.pi * freq * sym_dur) % (2 * np.pi)
        parts.append(chunk)

    sig = np.concatenate(parts)
    sig = _fade(sig, ms=20)
    return _normalise(sig)


# ── 5. Radio beacon  (two-tone ID + "DE" in Morse) ───────────────────────────
def make_radio_beacon() -> np.ndarray:
    # Two-tone callsign ID
    id_dur = 0.35
    ident  = (_sine(1500, id_dur) + _sine(900, id_dur)) * 0.5
    ident  = _fade(ident, ms=15)

    # "DE" in Morse: D = -.. | E = .
    freq  = 1000.0
    dot   = 0.08
    dash  = 0.24
    eg    = 0.08
    lg    = 0.28

    def dit(): return _fade(_sine(freq, dot))
    def dah(): return _fade(_sine(freq, dash))

    D = np.concatenate([dah(), _silence(eg), dit(), _silence(eg), dit()])
    E = dit()
    morse = np.concatenate([D, _silence(lg), E])

    sig = np.concatenate([ident, _silence(0.15), morse,
                          _silence(0.4),
                          ident, _silence(0.15), morse,
                          _silence(0.5)])
    return _normalise(sig)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    files = {
        "morse_sos.wav":    make_morse_sos(),
        "voice_male.wav":   make_voice_male(),
        "voice_female.wav": make_voice_female(),
        "data_afsk.wav":    make_data_afsk(),
        "radio_beacon.wav": make_radio_beacon(),
    }
    for fname, samples in files.items():
        save_wav(fname, samples)
    print("Done.")
