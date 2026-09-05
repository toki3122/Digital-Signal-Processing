import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
Fs = 1000
Fc = 150
order = 60
numtaps = order + 1
b = signal.firwin(
 numtaps,
 Fc,
 window='blackman',
 pass_zero='highpass',
 fs=Fs
)
f, h = signal.freqz(b, worN=4096, fs=Fs)
magnitude = 20*np.log10(np.maximum(abs(h), 1e-10))
phase = np.unwrap(np.angle(h))
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(f, magnitude)
plt.axvline(Fc, linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('60th Order Blackman High-Pass FIR Filter')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(f, phase)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (rad)')
plt.title('Phase Response')
plt.grid()
plt.tight_layout()
plt.show()