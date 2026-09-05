import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, kaiserord
Fs = 20000
fp = 1000
fstop = 4000
Rp = 0.25
As = 42
transition = fstop - fp
width = transition / (Fs/2)
N, beta = kaiserord(As, width)
if N % 2 == 0:
    N += 1
fc = (fp + fstop) / 2
h = firwin(N,fc,window=('kaiser', beta),pass_zero='lowpass',fs=Fs)
print("Number of taps =", N)
print("Kaiser beta =", beta)
print("Cutoff frequency =", fc, "Hz")
print("\nFilter coefficients:")
print(h)
w, H = freqz(h,worN=4096,fs=Fs)
plt.figure(figsize=(9, 5))
plt.subplot(2, 1, 1)
plt.plot(w, 20*np.log10(np.maximum(abs(H), 1e-8)))
plt.axvline(fp, linestyle='--')
plt.axvline(fstop, linestyle='--')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("FIR Lowpass Filter using Kaiser Window")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(w, np.unwrap(np.angle(H)))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (rad)")
plt.title("Phase Response")
plt.grid()
plt.tight_layout()
plt.show() 