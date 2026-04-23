FS        = 500_000      # Sample rate (Hz)
N_FFT     = 4096         # FFT window size → ~122 Hz/bin
CHUNK     = 512          # Samples per UDP packet
F_LO      = 40_000       # Display window low edge (Hz)
F_HI      = 240_000      # Display window high edge (Hz)
NOISE_STD = 0.015        # AWGN amplitude

UDP_HOST  = "127.0.0.1"
UDP_PORT  = 12345

COLORS = ["#00BFFF", "#FF6B6B", "#00FF9F", "#FFD700", "#BF5FFF"]
