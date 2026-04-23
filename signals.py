"""
Signal model — base class + AM, FM, ASK concrete types.
GUI interacts with all signals through SignalBase only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
import numpy as np

from config import FS

TWO_PI = 2 * np.pi

# Deterministic pseudo-random bit sequence used by ASK (seed fixed so every
# run produces the same pattern; long enough to avoid visible repetition)
_PRBS = np.random.RandomState(0xA5).randint(0, 2, 65536)


# ── Form metadata ─────────────────────────────────────────────────────────────
@dataclass
class FieldSpec:
    key:     str
    label:   str
    default: str


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


# ── Defaults ──────────────────────────────────────────────────────────────────
def default_signals() -> list[SignalBase]:
    return [
        AMSignal( "CH1",  60_000,  1_000, 0.50),
        AMSignal( "CH2",  90_000,  2_500, 0.70),
        AMSignal( "CH3", 130_000,  5_000, 0.40),
        FMSignal( "CH4", 170_000, 10_000, 8_000),
        ASKSignal("CH5", 210_000,  2_000, 0.0),
    ]
