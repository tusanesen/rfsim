"""
AM Signal Transmitter
Generates composite AM signal and streams chunks over UDP at real-time pace.
Run this first, then start signal_capture.py.
"""

import socket
import time
import numpy as np
from config import FS, CHUNK, NOISE_STD, SIGNALS, UDP_HOST, UDP_PORT

PACKET_BYTES = CHUNK * 4   # float32 → 4 bytes per sample


def generate_chunk(t_offset: float) -> np.ndarray:
    t = np.arange(CHUNK) / FS + t_offset
    composite = np.zeros(CHUNK)
    for fc, fm, m, _ in SIGNALS:
        composite += (1 + m * np.cos(2 * np.pi * fm * t)) * np.cos(2 * np.pi * fc * t)
    composite += np.random.randn(CHUNK) * NOISE_STD
    return composite.astype(np.float32)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (UDP_HOST, UDP_PORT)

    print(f"Transmitting to {UDP_HOST}:{UDP_PORT}  "
          f"[Fs={FS/1e3:.0f} kHz  chunk={CHUNK}  {PACKET_BYTES} bytes/packet]")
    print("Press Ctrl+C to stop.\n")

    t_offset = 0.0
    t_next   = time.perf_counter()

    try:
        while True:
            chunk = generate_chunk(t_offset)
            t_offset += CHUNK / FS

            sock.sendto(chunk.tobytes(), dest)

            # pace transmission to match real sample rate
            t_next += CHUNK / FS
            sleep_s = t_next - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nTransmitter stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
