# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
# Install dependencies into the local venv (first time only)
.venv/Scripts/python.exe -m pip install numpy matplotlib

# Start the signal generator (opens tkinter GUI + begins UDP transmit)
.venv/Scripts/python.exe signal_gen.py

# Start the spectrum visualizer (in a second terminal)
.venv/Scripts/python.exe signal_capture.py
```

Start `signal_gen.py` first so the UDP socket is transmitting before the capture side binds.

## Architecture

The project is split into two independent processes that communicate over loopback UDP.

**signal_gen.py** — tkinter GUI + transmitter thread. A background thread generates `CHUNK`-sized **complex64** numpy arrays at real-time pace (`time.perf_counter`-paced sleep) and sends them via UDP. The GUI modifies the shared `signals` list (protected by `threading.Lock`) and `noise_ref` in real time; the transmitter snapshots the list each chunk. A once-per-second benchmark line prints chunks/s and sleep budget so dropped pacing is visible.

**signal_capture.py** — matplotlib + receiver thread. A daemon thread receives UDP packets into a `queue.Queue`. A tk `after()` poll callback (main thread) drains the queue, rolls a ring buffer of `N_FFT` complex samples, runs a Blackman-windowed `fft + fftshift`, and updates the spectrum/time-domain artists.

**config.py** — single source of truth for all constants shared between the two processes: `FS`, `N_FFT`, `CHUNK`, baseband display bounds (`F_LO`, `F_HI`), `DISPLAY_OFFSET_HZ`, UDP endpoint. Both scripts import from here.

**am_simulation.py** — legacy standalone (static PNG output, no UDP). Superseded by `signal_gen.py`.

## Key design constraints

- **Packet format**: raw `complex64` IQ bytes, exactly `CHUNK * 8` bytes per datagram. Both sides must agree on `CHUNK` (via `config.py`). At the default `FS=8_000_000` and `CHUNK=4096`, that's ~1950 packets/s and ~64 MB/s on loopback.
- **Baseband convention**: signals are generated in baseband Hz with **signed** carrier frequencies (`-FS/2 < fc < +FS/2`). The receiver UI applies a fixed cosmetic offset of `DISPLAY_OFFSET_HZ` (default +150 MHz) so the displayed band reads e.g. 146-154 MHz while internal math stays at baseband. Only axis ticks, labels, and freq-entry parsing apply the offset; DDC/decimation/recording paths use baseband Hz throughout.
- **Phase continuity**: signal generation uses `t = np.arange(CHUNK)/FS + t_offset` where `t_offset` advances by `CHUNK/FS` each chunk. Carriers are emitted as `exp(2j*pi*fc*t)` (complex), so phase is continuous across packet boundaries without state.
- **Ring buffer overlap**: `signal_capture.py` shifts `CHUNK` new complex samples into an `N_FFT`-sample buffer each frame, giving smooth spectral motion without recomputing the full window from scratch.
- **tkinter + threading**: `noise_ref` is a `[float]` list (CPython GIL makes single-slot writes atomic). The signals list uses an explicit `Lock`; the tx thread always snapshots via `list(signals)` inside the lock before generating.
- **Recordings**: DDR captures are written as raw `complex64` IQ to `recordings/IQ_<fc_kHz>kHz_SR<sr_kHz>kHz_<timestamp>.bin`. `fc_kHz` may be **signed** (negative carriers in baseband). Folder is capped at 10 GB; oldest files evicted on startup.
- **Windows console encoding**: avoid non-ASCII characters (Unicode bullets, Greek letters) in any string that gets printed to stdout — the default Windows cp1252 codec will raise `UnicodeEncodeError`.

## Adding a new modulation type

Modulations live in `signals.py` as subclasses of `SignalBase`, registered via `@_register`. Each `generate(t)` must return `complex64` baseband (use `exp(2j*pi*fc*t)`, never `cos`). To add a type:
1. Subclass `SignalBase`, set `mod_type = "MYMOD"`, implement `type_field_specs()`, `from_form()`, `to_form()`, `validate()` (call `super().validate()`), `generate()`, `describe()`.
2. Decorate with `@_register` so the GUI radio group and the DDR demod selector pick it up automatically.
3. If demodulation is needed in the receiver, add a branch in `DDReceiver._demodulate` (`signal_capture.py`).
