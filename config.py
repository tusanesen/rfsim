FS                = 8_000_000       # 8 MHz IQ sample rate (complex64)
N_FFT             = 4096            # ~1.95 kHz/bin at FS=8 MHz
CHUNK             = 4096            # 32 KB packets (CHUNK * 8 bytes complex64)
F_LO              = -4_000_000      # baseband display low edge  (signed Hz)
F_HI              =  4_000_000      # baseband display high edge (signed Hz)
DISPLAY_OFFSET_HZ = 150_000_000     # cosmetic: f_display = f_baseband + 150 MHz
NOISE_STD         = 0.015           # complex AWGN amplitude

UDP_HOST  = "127.0.0.1"
UDP_PORT  = 12345

COLORS = ["#00BFFF", "#FF6B6B", "#00FF9F", "#FFD700", "#BF5FFF"]
