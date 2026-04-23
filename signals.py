"""
Signal model — base class + AM, FM, ASK, WAV-AM, WAV-FM concrete types.
GUI interacts with all signals through SignalBase only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
import os
import wave
import numpy as np

from config import FS

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")

TWO_PI = 2 * np.pi

# Deterministic pseudo-random bit sequence used by ASK (seed fixed so every
# run produces the same pattern; long enough to avoid visible repetition)
_PRBS = np.random.RandomState(0xA5).randint(0, 2, 65536)


# ── WAV helpers ───────────────────────────────────────────────────────────────
def _list_wav_files() -> list:
    if not os.path.isdir(AUDIO_DIR):
        return []
    return sorted(f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav"))


def _load_wav(filename: str) -> tuple:
    """Load a WAV from AUDIO_DIR. Returns (samples: float32 ndarray, sample_rate: int)."""
    path = os.path.join(AUDIO_DIR, filename)
    with wave.open(path, "r") as wf:
        nch  = wf.getnchannels()
        sw   = wf.getsampwidth()
        rate = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())

    if sw == 2:
        s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        s = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

    if nch > 1:
        s = s.reshape(-1, nch).mean(axis=1)

    peak = np.abs(s).max()
    if peak > 0:
        s /= peak

    return s.astype(np.float32), rate


# ── Form metadata ─────────────────────────────────────────────────────────────
@dataclass
class FieldSpec:
    key:     str
    label:   str
    default: str
    choices: list = None   # if set → rendered as a Combobox instead of Entry


# ── Base class ────────────────────────────────────────────────────────────────
class SignalBase(ABC):
    mod_type: ClassVar[str] = ""

    def __init__(self, label: str, fc: float, amplitude: float, enabled: bool = True):
        self.label     = label
        self.fc        = fc
        self.amplitude = amplitude
        self.enabled   = enabled

    # ── Form metadata (class-level) ───────────────────────────────────────────
    @classmethod
    def common_field_specs(cls) -> list[FieldSpec]:
        return [
            FieldSpec("label", "Label",         "SIG"),
            FieldSpec("fc",    "Carrier (kHz)", "100"),
            FieldSpec("amp",   "Amplitude",     "1.0"),
        ]

    @classmethod
    @abstractmethod
    def type_field_specs(cls) -> list[FieldSpec]:
        """Return field specs for this modulation's parameters (no common fields)."""
        ...

    # ── Serialisation to/from GUI form dict ───────────────────────────────────
    @classmethod
    @abstractmethod
    def from_form(cls, values: dict[str, str]) -> "SignalBase":
        """Construct an instance from a flat string-value dict (no validation)."""
        ...

    @abstractmethod
    def to_form(self) -> dict[str, str]:
        """Return a dict that can pre-fill the GUI form for this signal."""
        ...

    # ── Validation ────────────────────────────────────────────────────────────
    @classmethod
    def validate(cls, values: dict[str, str]) -> list[str]:
        """Return a list of human-readable error strings (empty = valid)."""
        errors = []
        nyq = FS / 2
        try:
            fc = float(values["fc"]) * 1000
            if not (0 < fc < nyq):
                errors.append(f"Carrier must be between 0 and {nyq/1000:.0f} kHz.")
        except (ValueError, KeyError):
            errors.append("Carrier must be a number.")
        try:
            amp = float(values["amp"])
            if amp <= 0:
                errors.append("Amplitude must be > 0.")
        except (ValueError, KeyError):
            errors.append("Amplitude must be a number.")
        return errors

    # ── Signal generation ─────────────────────────────────────────────────────
    @abstractmethod
    def generate(self, t: np.ndarray) -> np.ndarray:
        """Return samples for the given time vector (phase-continuous)."""
        ...

    # ── List display ──────────────────────────────────────────────────────────
    @abstractmethod
    def describe(self) -> str:
        """One-line string for the GUI listbox."""
        ...

    def _tag(self) -> str:
        return "[on] " if self.enabled else "[off]"


# ── Registry ──────────────────────────────────────────────────────────────────
SIGNAL_REGISTRY: dict[str, type[SignalBase]] = {}

def _register(cls: type) -> type:
    SIGNAL_REGISTRY[cls.mod_type] = cls
    return cls


# ── AM ────────────────────────────────────────────────────────────────────────
@_register
class AMSignal(SignalBase):
    mod_type = "AM"

    def __init__(self, label, fc, fm, mod_index, amplitude=1.0, enabled=True):
        super().__init__(label, fc, amplitude, enabled)
        self.fm        = fm
        self.mod_index = mod_index

    @classmethod
    def type_field_specs(cls) -> list[FieldSpec]:
        return [
            FieldSpec("fm",        "Mod Freq (kHz)", "2"),
            FieldSpec("mod_index", "Mod Index (0-1)", "0.5"),
        ]

    @classmethod
    def from_form(cls, v: dict) -> "AMSignal":
        return cls(
            label     = v["label"].strip() or "SIG",
            fc        = float(v["fc"])        * 1000,
            fm        = float(v["fm"])        * 1000,
            mod_index = float(v["mod_index"]),
            amplitude = float(v["amp"]),
        )

    @classmethod
    def validate(cls, v: dict) -> list[str]:
        errors = super().validate(v)
        try:
            fm = float(v["fm"]) * 1000
            if fm <= 0:
                errors.append("Mod frequency must be > 0.")
        except (ValueError, KeyError):
            errors.append("Mod frequency must be a number.")
        try:
            m = float(v["mod_index"])
            if not (0 < m <= 1.0):
                errors.append("Mod index must be in (0, 1].")
        except (ValueError, KeyError):
            errors.append("Mod index must be a number.")
        return errors

    def to_form(self) -> dict:
        return {
            "label": self.label,        "fc":  f"{self.fc/1000:.3f}",
            "amp":   f"{self.amplitude:.2f}",
            "fm":    f"{self.fm/1000:.4f}", "mod_index": f"{self.mod_index:.3f}",
        }

    def generate(self, t: np.ndarray) -> np.ndarray:
        return (self.amplitude
                * (1.0 + self.mod_index * np.cos(TWO_PI * self.fm * t))
                * np.cos(TWO_PI * self.fc * t))

    def describe(self) -> str:
        return (f"{self._tag()} [AM]  {self.label:<6}  "
                f"fc={self.fc/1000:.1f}kHz  fm={self.fm/1000:.2f}kHz  "
                f"m={self.mod_index:.2f}  A={self.amplitude:.2f}")


# ── FM ────────────────────────────────────────────────────────────────────────
@_register
class FMSignal(SignalBase):
    mod_type = "FM"

    def __init__(self, label, fc, fm, deviation, amplitude=1.0, enabled=True):
        super().__init__(label, fc, amplitude, enabled)
        self.fm        = fm
        self.deviation = deviation   # Hz

    @classmethod
    def type_field_specs(cls) -> list[FieldSpec]:
        return [
            FieldSpec("fm",        "Mod Freq (kHz)", "5"),
            FieldSpec("deviation", "Deviation (Hz)", "5000"),
        ]

    @classmethod
    def from_form(cls, v: dict) -> "FMSignal":
        return cls(
            label     = v["label"].strip() or "SIG",
            fc        = float(v["fc"])        * 1000,
            fm        = float(v["fm"])        * 1000,
            deviation = float(v["deviation"]),
            amplitude = float(v["amp"]),
        )

    @classmethod
    def validate(cls, v: dict) -> list[str]:
        errors = super().validate(v)
        try:
            fm = float(v["fm"]) * 1000
            if fm <= 0:
                errors.append("Mod frequency must be > 0.")
        except (ValueError, KeyError):
            errors.append("Mod frequency must be a number.")
        try:
            d = float(v["deviation"])
            if d <= 0:
                errors.append("Deviation must be > 0 Hz.")
        except (ValueError, KeyError):
            errors.append("Deviation must be a number.")
        return errors

    def to_form(self) -> dict:
        return {
            "label": self.label,        "fc":  f"{self.fc/1000:.3f}",
            "amp":   f"{self.amplitude:.2f}",
            "fm":    f"{self.fm/1000:.4f}", "deviation": f"{self.deviation:.1f}",
        }

    def generate(self, t: np.ndarray) -> np.ndarray:
        beta = self.deviation / self.fm
        return self.amplitude * np.cos(TWO_PI * self.fc * t
                                       + beta * np.sin(TWO_PI * self.fm * t))

    def describe(self) -> str:
        return (f"{self._tag()} [FM]  {self.label:<6}  "
                f"fc={self.fc/1000:.1f}kHz  fm={self.fm/1000:.2f}kHz  "
                f"dev={self.deviation/1000:.1f}kHz  A={self.amplitude:.2f}")


# ── ASK ───────────────────────────────────────────────────────────────────────
@_register
class ASKSignal(SignalBase):
    """
    2-ASK / OOK.  Amplitude toggles between `amplitude` (bit=1) and
    `amplitude * low_level` (bit=0) at the given bitrate.
    low_level=0 → OOK (on-off keying).
    Bit pattern is deterministic (seeded PRBS) so phase is reproducible
    from t_offset alone — no state needed across chunk boundaries.
    """
    mod_type = "ASK"

    def __init__(self, label, fc, bitrate, low_level=0.0, amplitude=1.0, enabled=True):
        super().__init__(label, fc, amplitude, enabled)
        self.bitrate   = bitrate     # bps
        self.low_level = low_level   # amplitude ratio for '0' bit

    @classmethod
    def type_field_specs(cls) -> list[FieldSpec]:
        return [
            FieldSpec("bitrate",   "Bit Rate (bps)", "2000"),
            FieldSpec("low_level", "Low Level (0-1)", "0.0"),
        ]

    @classmethod
    def from_form(cls, v: dict) -> "ASKSignal":
        return cls(
            label     = v["label"].strip() or "SIG",
            fc        = float(v["fc"])        * 1000,
            bitrate   = float(v["bitrate"]),
            low_level = float(v["low_level"]),
            amplitude = float(v["amp"]),
        )

    @classmethod
    def validate(cls, v: dict) -> list[str]:
        errors = super().validate(v)
        try:
            br = float(v["bitrate"])
            if br <= 0:
                errors.append("Bit rate must be > 0.")
        except (ValueError, KeyError):
            errors.append("Bit rate must be a number.")
        try:
            lo = float(v["low_level"])
            if not (0.0 <= lo < 1.0):
                errors.append("Low level must be in [0, 1).")
        except (ValueError, KeyError):
            errors.append("Low level must be a number.")
        return errors

    def to_form(self) -> dict:
        return {
            "label": self.label,        "fc":  f"{self.fc/1000:.3f}",
            "amp":   f"{self.amplitude:.2f}",
            "bitrate":   f"{self.bitrate:.0f}",
            "low_level": f"{self.low_level:.2f}",
        }

    def generate(self, t: np.ndarray) -> np.ndarray:
        bit_idx  = (t * self.bitrate).astype(np.int64) % len(_PRBS)
        bits     = _PRBS[bit_idx]
        envelope = np.where(bits, self.amplitude, self.amplitude * self.low_level)
        return envelope * np.cos(TWO_PI * self.fc * t)

    def describe(self) -> str:
        mode = "OOK" if self.low_level == 0.0 else "2-ASK"
        return (f"{self._tag()} [ASK] {self.label:<6}  "
                f"fc={self.fc/1000:.1f}kHz  br={self.bitrate:.0f}bps  "
                f"lo={self.low_level:.2f} ({mode})  A={self.amplitude:.2f}")


# ── WAV-AM ────────────────────────────────────────────────────────────────────
@_register
class WavAMSignal(SignalBase):
    """AM signal using a WAV file as the modulating waveform instead of a sine."""
    mod_type = "WAV-AM"

    def __init__(self, label, fc, wav_file, depth, amplitude=1.0, enabled=True):
        super().__init__(label, fc, amplitude, enabled)
        self.depth = depth
        self._set_wav(wav_file)

    def _set_wav(self, wav_file: str):
        self.wav_file = wav_file
        path = os.path.join(AUDIO_DIR, wav_file) if wav_file else ""
        if wav_file and os.path.isfile(path):
            self._samples, self._audio_fs = _load_wav(wav_file)
        else:
            # fallback: 1 kHz sine at 8 kHz
            self._audio_fs = 8_000
            t = np.arange(self._audio_fs) / self._audio_fs
            self._samples = np.sin(TWO_PI * 1000 * t).astype(np.float32)

    @classmethod
    def type_field_specs(cls) -> list:
        wavs = _list_wav_files()
        return [
            FieldSpec("wav_file", "Audio File",    wavs[0] if wavs else "", choices=wavs),
            FieldSpec("depth",    "Mod Depth (0-1)", "0.7"),
        ]

    @classmethod
    def from_form(cls, v: dict) -> "WavAMSignal":
        return cls(
            label     = v["label"].strip() or "SIG",
            fc        = float(v["fc"]) * 1000,
            wav_file  = v.get("wav_file", ""),
            depth     = float(v["depth"]),
            amplitude = float(v["amp"]),
        )

    @classmethod
    def validate(cls, v: dict) -> list:
        errors = super().validate(v)
        if not v.get("wav_file"):
            errors.append("Select an audio file.")
        try:
            d = float(v["depth"])
            if not (0 < d <= 1.0):
                errors.append("Mod depth must be in (0, 1].")
        except (ValueError, KeyError):
            errors.append("Mod depth must be a number.")
        return errors

    def to_form(self) -> dict:
        return {
            "label": self.label, "fc": f"{self.fc/1000:.3f}",
            "amp": f"{self.amplitude:.2f}",
            "wav_file": self.wav_file, "depth": f"{self.depth:.2f}",
        }

    def generate(self, t: np.ndarray) -> np.ndarray:
        idx = (t * self._audio_fs).astype(np.int64) % len(self._samples)
        m   = self._samples[idx]
        return self.amplitude * (1.0 + self.depth * m) * np.cos(TWO_PI * self.fc * t)

    def describe(self) -> str:
        return (f"{self._tag()} [WAV-AM] {self.label:<6}  "
                f"fc={self.fc/1000:.1f}kHz  [{self.wav_file}]  "
                f"depth={self.depth:.2f}  A={self.amplitude:.2f}")


# ── WAV-FM ────────────────────────────────────────────────────────────────────
@_register
class WavFMSignal(SignalBase):
    """
    FM signal using a WAV file as the modulating waveform.
    Phase integral is pre-computed as a cumulative sum so generate() is
    stateless: any t vector produces the correct continuous phase.
    Loop discontinuity is handled by accumulating whole-loop integrals.
    """
    mod_type = "WAV-FM"

    def __init__(self, label, fc, wav_file, deviation, amplitude=1.0, enabled=True):
        super().__init__(label, fc, amplitude, enabled)
        self.deviation = deviation
        self._set_wav(wav_file)

    def _set_wav(self, wav_file: str):
        self.wav_file = wav_file
        path = os.path.join(AUDIO_DIR, wav_file) if wav_file else ""
        if wav_file and os.path.isfile(path):
            self._samples, self._audio_fs = _load_wav(wav_file)
        else:
            self._audio_fs = 8_000
            t = np.arange(self._audio_fs) / self._audio_fs
            self._samples = np.sin(TWO_PI * 1000 * t).astype(np.float32)
        # Pre-compute cumulative integral of the modulator (used for FM phase)
        self._cumsum        = np.cumsum(self._samples).astype(np.float64) / self._audio_fs
        self._loop_integral = float(self._cumsum[-1])

    @classmethod
    def type_field_specs(cls) -> list:
        wavs = _list_wav_files()
        return [
            FieldSpec("wav_file",  "Audio File",     wavs[0] if wavs else "", choices=wavs),
            FieldSpec("deviation", "Deviation (Hz)", "5000"),
        ]

    @classmethod
    def from_form(cls, v: dict) -> "WavFMSignal":
        return cls(
            label     = v["label"].strip() or "SIG",
            fc        = float(v["fc"]) * 1000,
            wav_file  = v.get("wav_file", ""),
            deviation = float(v["deviation"]),
            amplitude = float(v["amp"]),
        )

    @classmethod
    def validate(cls, v: dict) -> list:
        errors = super().validate(v)
        if not v.get("wav_file"):
            errors.append("Select an audio file.")
        try:
            d = float(v["deviation"])
            if d <= 0:
                errors.append("Deviation must be > 0 Hz.")
        except (ValueError, KeyError):
            errors.append("Deviation must be a number.")
        return errors

    def to_form(self) -> dict:
        return {
            "label": self.label, "fc": f"{self.fc/1000:.3f}",
            "amp": f"{self.amplitude:.2f}",
            "wav_file": self.wav_file, "deviation": f"{self.deviation:.1f}",
        }

    def generate(self, t: np.ndarray) -> np.ndarray:
        n          = len(self._samples)
        idx        = (t * self._audio_fs).astype(np.int64)
        full_loops = idx // n
        within_idx = idx % n
        # ∫₀ᵗ m(τ)dτ = integral of k full loops + integral within current loop
        phase_int  = self._cumsum[within_idx] + full_loops * self._loop_integral
        return self.amplitude * np.cos(TWO_PI * self.fc * t
                                        + TWO_PI * self.deviation * phase_int)

    def describe(self) -> str:
        return (f"{self._tag()} [WAV-FM] {self.label:<6}  "
                f"fc={self.fc/1000:.1f}kHz  [{self.wav_file}]  "
                f"dev={self.deviation/1000:.1f}kHz  A={self.amplitude:.2f}")


# ── Defaults ──────────────────────────────────────────────────────────────────
def default_signals() -> list[SignalBase]:
    wavs = _list_wav_files()
    defaults = [
        AMSignal( "CH1",  60_000,  1_000, 0.50),
        AMSignal( "CH2",  90_000,  2_500, 0.70),
        AMSignal( "CH3", 130_000,  5_000, 0.40),
        FMSignal( "CH4", 170_000, 10_000, 8_000),
        ASKSignal("CH5", 210_000,  2_000, 0.0),
    ]
    if wavs:
        defaults.append(WavAMSignal("CH6", 80_000,  wavs[0], 0.85))
        defaults.append(WavFMSignal("CH7", 150_000, wavs[1] if len(wavs) > 1 else wavs[0], 6_000))
    return defaults
