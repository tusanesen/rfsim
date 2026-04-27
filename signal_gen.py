"""
Signal Generator — entry point.
Starts the UDP transmitter thread, then launches the tkinter GUI.
Wire format: complex64 IQ samples, CHUNK*8 bytes per UDP datagram.
"""

import socket
import threading
import time
import numpy as np

from config import FS, CHUNK, NOISE_STD, UDP_HOST, UDP_PORT
from signals import default_signals
from signal_gui import App

PACKET_BYTES = CHUNK * 8     # complex64 = 8 bytes/sample
_NOISE_SCALE = 1.0 / np.sqrt(2.0)   # split power across I and Q


def _tx_worker(stop_evt, signals, sig_lock, noise_ref, pkt_counter):
    sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest   = (UDP_HOST, UDP_PORT)
    t_off  = 0.0
    t_next = time.perf_counter()

    bench_t0       = time.perf_counter()
    bench_chunks   = 0
    bench_sleep_s  = 0.0
    bench_late     = 0
    bench_min_slp  = float("inf")

    while not stop_evt.is_set():
        with sig_lock:
            snap = [s for s in signals if s.enabled]

        t   = np.arange(CHUNK, dtype=np.float64) / FS + t_off
        out = np.zeros(CHUNK, dtype=np.complex64)
        for s in snap:
            out += s.generate(t)

        n = noise_ref[0]
        if n > 0.0:
            noise = (np.random.randn(CHUNK) + 1j * np.random.randn(CHUNK)) * (n * _NOISE_SCALE)
            out  += noise.astype(np.complex64)

        t_off += CHUNK / FS

        try:
            sock.sendto(out.tobytes(), dest)
            pkt_counter[0] += 1
            bench_chunks   += 1
        except OSError:
            pass

        t_next += CHUNK / FS
        sleep_s = t_next - time.perf_counter()
        if sleep_s > 0:
            bench_sleep_s += sleep_s
            if sleep_s < bench_min_slp:
                bench_min_slp = sleep_s
            time.sleep(sleep_s)
        else:
            bench_late += 1

        now = time.perf_counter()
        if now - bench_t0 >= 1.0:
            avg_slp_us = (bench_sleep_s / max(bench_chunks, 1)) * 1e6
            min_slp_us = bench_min_slp * 1e6 if bench_min_slp != float("inf") else 0.0
            print(f"tx: {bench_chunks} chunks/s  "
                  f"sleep avg={avg_slp_us:.1f}us min={min_slp_us:.1f}us  "
                  f"late={bench_late}")
            bench_t0      = now
            bench_chunks  = 0
            bench_sleep_s = 0.0
            bench_late    = 0
            bench_min_slp = float("inf")

    sock.close()


def main():
    signals     = default_signals()
    sig_lock    = threading.Lock()
    noise_ref   = [NOISE_STD]
    stop_evt    = threading.Event()
    pkt_counter = [0]

    tx = threading.Thread(
        target=_tx_worker,
        args=(stop_evt, signals, sig_lock, noise_ref, pkt_counter),
        daemon=True,
    )
    tx.start()

    App(signals, sig_lock, noise_ref, pkt_counter, stop_evt).mainloop()
    stop_evt.set()


if __name__ == "__main__":
    main()
