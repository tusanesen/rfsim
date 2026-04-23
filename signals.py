"""
Signal model — base class + AM, FM, ASK concrete types.
AM and FM each carry an optional WAV modulator; when unset they fall back
to a sine wave so the behaviour is identical to before.
GUI interacts with all signals through SignalBase only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
import os
import wave
import numpy as np

from config import FS

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
_WAV_DEFAULT = "(default)"   # sentinel meaning "use sine modulator"

TWO_PI = 2 * np.pi

# Deterministic PRBS for ASK
_PRBS = np.random.RandomState(0xA5).randint(0, 2, 65536)


# ── WAV helpers ───────────────────────────────────────────────────────────────
def _list_wav_files() -> list:
    if not os.path.isdir(AUDIO_DIR):
        return []
    return sorted(f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav"))


def _wav_choices() -> list:
    return [_WAV_DEFAULT] + _list_wav_files()


def _load_wav(filename: str) -> tuple:
    """Returns (samples: float32 ndarray normalised to ±1, sample_rate: int)."""
    path = os.path.join(AUDIO_DIR, filename)
    with wave.open(path, "r") as wf:
        nch = wf.getnchannels()
        sw  = wf.getsampwidth()
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

    @classmethod
    def common_field_specs(cls) -> list:
        return [
            FieldSpec("label", "Label",         "SIG"),
            FieldSpec("fc",    "Carrier (kHz)", "100"),
            FieldSpec("amp",   "Amplitude",     "1.0"),
        ]

    @classmethod
    @abstractmethod
    def type_field_specs(cls) -> list:
        ...

    @classmethod
    @abstractmethod
    def from_form(cls, values: dict) -> "SignalBase":
        ...

    @abstractmethod
    def to_form(self) -> dict:
        ...

    @classmethod
    def validate(cls, values: dict) -> list:
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

    @abstractmethod
    def generate(self, t: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...

    def _tag(self) -> str:
        return "[on] " if self.enabled else "[off]"


# ── Registry ──────────────────────────────────────────────────────────────────
SIGNAL_REGISTRY: dict = {}

def _register(cls: type) -> type:
    SIGNAL_REGISTRY[cls.mod_type] = cls
    return cls


# ── AM ────────────────────────────────────────────────────────────────────────
@_register
class AMSignal(SignalBase):
    mod_type = "AM"

    def __init__(self, label, fc, fm, mod_index, amplitude=1.0,
                 wav_file=_WAV_DEFAULT, enabled=True):
        super().__init__(label, fc, amplitude, enabled)
        self.fm        = fm
        self.mod_index = mod_index
        self._load_wav(wav_file)

    def _load_wav(self, wav_file: str):
        self.wav_file  = wav_file
        self._samples  = None
        self._audio_fs = 8_000
        if wav_file and wav_file != _WAV_DEFAULT:
            try:
                self._samples, self._audio_fs = _load_wav(wav_file)
            except Exception:
                self._samples = None

    @classmethod
    def type_field_specs(cls) -> list:
        return [
            FieldSpec("fm",        "Mod Freq (kHz)",  "2"),
            FieldSpec("mod_index", "Mod Index (0-1)", "0.5"),
            FieldSpec("wav_file",  "Modulator",       _WAV_DEFAULT, choices=_wav_choices()),
        ]

    @classmethod
    def from_form(cls, v: dict) -> "AMSignal":
        return cls(
            label     = v["label"].strip() or "SIG",
            fc        = float(v["fc"])        * 1000,
            fm        = float(v["fm"])        * 1000,
            mod_index = float(v["mod_index"]),
            amplitude = float(v["amp"]),
            wav_file  = v.get("wav_file", _WAV_DEFAULT),
        )

    @classmethod
    def validate(cls, v: dict) -> list:
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
            "label":     self.label,
            "fc":        f"{self.fc/1000:.3f}",
            "amp":       f"{self.amplitude:.2f}",
            "fm":        f"{self.fm/1000:.4f}",
            "mod_index": f"{self.mod_index:.3f}",
            "wav_file":  self.wav_file,
        }

    def generate(self, t: np.ndarray) -> np.ndarray:
        if self._samples is not None:
            idx = (t * self._audio_fs).astype(np.int64) % len(self._samples)
            m   = self._samples[idx]
        else:
            m = np.cos(TWO_PI * self.fm * t)
        return self.amplitude * (1.0 + self.mod_index * m) * np.cos(TWO_PI * self.fc * t)

    def describe(self) -> str:
        wav = f"  [{self.wav_file}]" if self._samples is not None else ""
        return (f"{self._tag()} [AM]  {self.label:<6}  "
                f"fc={self.fc/1000:.1f}kHz  fm={self.fm/1000:.2f}kHz  "
                f"m={self.mod_index:.2f}  A={self.amplitude:.2f}{wav}")


# ── FM ────────────────────────────────────────────────────────────────────────
@_register
class FMSignal(SignalBase):
    mod_type = "FM"

    def __init__(self, label, fc, fm, deviation, amplitude=1.0,
                 wav_file=_WAV_DEFAULT, enabled=True):
        super().__init__(label, fc, amplitude, enabled)
        self.fm        = fm
        self.deviation = deviation
        self._load_wav(wav_file)

    def _load_wav(self, wav_file: str):
        self.wav_file       = wav_file
        self._samples       = None
        self._audio_fs      = 8_000
        self._cumsum        = None
        self._loop_integral = 0.0
        if wav_file and wav_file != _WAV_DEFAULT:
            try:
                self._samples, self._audio_fs = _load_wav(wav_file)
                # Pre-compute phase integral for stateless FM generation across chunks
                self._cumsum        = np.cumsum(self._samples).astype(np.float64) / self._audio_fs
                self._loop_integral = float(self._cumsum[-1])
            except Exception:
                self._samples = None

    @classmethod
    def type_field_specs(cls) -> list:
        return [
            FieldSpec("fm",        "Mod Freq (kHz)", "5"),
            FieldSpec("deviation", "Deviation (Hz)", "5000"),
            FieldSpec("wav_file",  "Modulator",      _WAV_DEFAULT, choices=_wav_choices()),
        ]

    @classmethod
    def from_form(cls, v: dict) -> "FMSignal":
        return cls(
            label     = v["label"].strip() or "SIG",
            fc        = float(v["fc"])        * 1000,
            fm        = float(v["fm"])        * 1000,
            deviation = float(v["deviation"]),
            amplitude = float(v["amp"]),
            wav_file  = v.get("wav_file", _WAV_DEFAULT),
        )

    @classmethod
    def validate(cls, v: dict) -> list:
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
            "label":     self.label,
            "fc":        f"{self.fc/1000:.3f}",
            "amp":       f"{self.amplitude:.2f}",
            "fm":        f"{self.fm/1000:.4f}",
            "deviation": f"{self.deviation:.1f}",
            "wav_file":  self.wav_file,
        }

    def generate(self, t: np.ndarray) -> np.ndarray:
        if self._samples is not None and self._cumsum is not None:
            n          = len(self._samples)
            idx        = (t * self._audio_fs).astype(np.int64)
            full_loops = idx // n
            within_idx = idx % n
            phase_int  = self._cumsum[within_idx] + full_loops * self._loop_integral
            return self.amplitude * np.cos(TWO_PI * self.fc * t
                                            + TWO_PI * self.deviation * phase_int)
        else:
            beta = self.deviation / self.fm
            return self.amplitude * np.cos(TWO_PI * self.fc * t
                                           + beta * np.sin(TWO_PI * self.fm * t))

    def describe(self) -> str:
        wav = f"  [{self.wav_file}]" if self._samples is not None else ""
        return (f"{self._tag()} [FM]  {self.label:<6}  "
                f"fc={self.fc/1000:.1f}kHz  fm={self.fm/1000:.2f}kHz  "
                f"dev={self.deviation/1000:.1f}kHz  A={self.amplitude:.2f}{wav}")


# ── ASK ───────────────────────────────────────────────────────────────────────
@_register
class ASKSignal(SignalBase):
    """
    2-ASK / OOK. Amplitude toggles between amplitude (bit=1) and
    amplitude*low_level (bit=0). low_level=0 → OOK.
    Bit pattern is deterministic from t alone — no state across chunks.
    """
    mod_type = "ASK"

    def __init__(self, label, fc, bitrate, low_level=0.0, amplitude=1.0, enabled=True):
        super().__init__(label, fc, amplitude, enabled)
        self.bitrate   = bitrate
        self.low_level = low_level

    @classmethod
    def type_field_specs(cls) -> list:
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
    def validate(cls, v: dict) -> list:
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
            "label":     self.label,
            "fc":        f"{self.fc/1000:.3f}",
            "amp":       f"{self.amplitude:.2f}",
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


# ── Defaults ──────────────────────────────────────────────────────────────────
def default_signals() -> list:
    wavs = _list_wav_files()
    return [
        AMSignal( "CH1",  60_000,  1_000, 0.50),
        AMSignal( "CH2",  90_000,  2_500, 0.70),
        AMSignal( "CH3", 130_000,  5_000, 0.40),
        FMSignal( "CH4", 170_000, 10_000, 8_000),
        ASKSignal("CH5", 210_000,  2_000, 0.0),
        AMSignal( "CH6",  80_000,  1_000, 0.85, wav_file=wavs[0] if wavs else _WAV_DEFAULT),
        FMSignal( "CH7", 150_000,  5_000, 6_000,
                  wav_file=wavs[1] if len(wavs) > 1 else _WAV_DEFAULT),
    ]
