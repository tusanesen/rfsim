FS        = 500_000      # Sample rate (Hz)
N_FFT     = 4096         # FFT window size → ~122 Hz/bin
CHUNK     = 512          # Samples per UDP packet
F_LO      = 40_000       # Display window low edge (Hz)
F_HI      = 240_000      # Display window high edge (Hz)
NOISE_STD = 0.015        # AWGN amplitude

UDP_HOST  = "127.0.0.1"
UDP_PORT  = 12345

SIGNALS = [
    ( 60_000,  1_000, 0.50, "CH1"),
    ( 90_000,  2_500, 0.70, "CH2"),
    (130_000,  5_000, 0.40, "CH3"),
    (170_000, 10_000, 0.80, "CH4"),
    (210_000,  3_000, 0.60, "CH5"),
]
COLORS = ["#00BFFF", "#FF6B6B", "#00FF9F", "#FFD700", "#BF5FFF"]
