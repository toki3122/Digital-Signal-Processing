import numpy as np
import matplotlib.pyplot as plt
Fs = 1000
f0 = 5
duration = 1
t = np.arange(0, duration, 1/Fs)
x = np.sign(np.sin(2 * np.pi * f0 * t))
X = np.fft.fft(x)
freq = np.fft.fftfreq(len(x), 1/Fs)
X_shift = np.fft.fftshift(X)
freq_shift = np.fft.fftshift(freq)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t[:400], x[:400])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("5 Hz Square Wave")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(freq_shift, np.abs(X_shift) / len(x))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectrum of 5 Hz Square Wave")
plt.xlim(-100, 100)
plt.grid()
plt.tight_layout()
plt.show() 