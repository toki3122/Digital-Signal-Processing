import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
Fs = 1000
passband = [150, 250]
tw = 50
attenuation = 50
numtaps, beta = signal.kaiserord(
 attenuation,
 tw / (Fs / 2)
)
print("Number of taps =", numtaps)
print("Kaiser beta =", beta)
b = signal.firwin(
 numtaps,
 passband,
 window=('kaiser', beta),
 pass_zero=False,
 fs=Fs
)
f, h = signal.freqz(b, worN=4096, fs=Fs)
magnitude = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
phase = np.unwrap(np.angle(h))
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(f, magnitude)
plt.axvline(150, linestyle='--')
plt.axvline(250, linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('FIR Band-Pass Filter - Magnitude Response')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(f, phase)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (rad)')
plt.title('FIR Band-Pass Filter - Phase Response')
plt.grid()
plt.tight_layout()
plt.show() 