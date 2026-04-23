"""
Signal Generator — entry point.
Starts the UDP transmitter thread, then launches the tkinter GUI.
"""

import socket
import threading
import time
import numpy as np

from config import FS, CHUNK, NOISE_STD, UDP_HOST, UDP_PORT
from signals import default_signals
from signal_gui import App

PACKET_BYTES = CHUNK * 4


def _tx_worker(stop_evt, signals, sig_lock, noise_ref, pkt_counter):
    sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest   = (UDP_HOST, UDP_PORT)
    t_off  = 0.0
    t_next = time.perf_counter()

    while not stop_evt.is_set():
        with sig_lock:
            snap = [s for s in signals if s.enabled]

        t   = np.arange(CHUNK, dtype=np.float64) / FS + t_off
        out = np.zeros(CHUNK, dtype=np.float64)
        for s in snap:
            out += s.generate(t)
        out += np.random.randn(CHUNK) * noise_ref[0]

        t_off += CHUNK / FS

        try:
            sock.sendto(out.astype(np.float32).tobytes(), dest)
            pkt_counter[0] += 1
        except OSError:
            pass

        t_next += CHUNK / FS
        sleep_s = t_next - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)

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
