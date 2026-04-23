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

**signal_gen.py** — tkinter GUI + transmitter thread. A background thread generates `CHUNK`-sized float32 numpy arrays at real-time pace (`time.perf_counter`-paced sleep) and sends them via UDP. The GUI modifies the shared `signals` list (protected by `threading.Lock`) and `noise_ref` in real time; the transmitter snapshots the list each chunk.

**signal_capture.py** — matplotlib animation + receiver thread. A daemon thread receives UDP packets into a `queue.Queue`. The `FuncAnimation` callback (main thread) drains the queue, rolls a ring buffer of `N_FFT` samples, runs a Blackman-windowed `rfft`, and updates two `Line2D` artists (spectrum + time-domain strip) with `blit=True`.

**config.py** — single source of truth for all constants shared between the two processes: `FS`, `N_FFT`, `CHUNK`, display window bounds, UDP endpoint. Both scripts import from here.

**am_simulation.py** — legacy standalone (static PNG output, no UDP). Superseded by `signal_gen.py`.

## Key design constraints

- **Packet format**: raw `float32` bytes, exactly `CHUNK * 4` bytes per datagram. Both sides must agree on `CHUNK` (via `config.py`).
- **Phase continuity**: signal generation uses `t = np.arange(CHUNK)/FS + t_offset` where `t_offset` advances by `CHUNK/FS` each chunk. This keeps carrier phase continuous across packet boundaries without any state accumulation.
- **Ring buffer overlap**: `signal_capture.py` shifts `CHUNK` new samples into an `N_FFT`-sample buffer each frame (87.5% overlap), giving smooth spectral motion without recomputing the full window from scratch.
- **blit=True + `cache_frame_data=False`**: both are required in `FuncAnimation` — blit for frame rate, `cache_frame_data=False` to prevent unbounded memory growth in long sessions.
- **tkinter + threading**: `noise_ref` is a `[float]` list (CPython GIL makes single-slot writes atomic). The signals list uses an explicit `Lock`; the tx thread always snapshots via `list(signals)` inside the lock before generating.
- **Windows console encoding**: avoid non-ASCII characters (Unicode bullets, Greek letters) in any string that gets printed to stdout — the default Windows cp1252 codec will raise `UnicodeEncodeError`.

## Adding a new modulation type

The `Signal` dataclass in `signal_gen.py` uses `mod_type` ("AM" / "FM") to branch in `generate_chunk()`. To add a type:
1. Add a branch in `generate_chunk()` for the new `mod_type` string.
2. Update the GUI form: add the new type to the radiobutton group and update `_on_type_change()` / `_param_lbl_var` logic.
3. Update `_parse_form()` validation for the new type's parameters.
