import numpy as np
import matplotlib.pyplot as plt
fs = 5000
N = 1024
n = np.arange(N)
x = np.sinc(n / 50)**2
X = np.fft.fft(x, N)
f = np.fft.fftfreq(N, 1/fs)
plt.plot(np.fft.fftshift(f),
 np.abs(np.fft.fftshift(X)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('DFT Spectrum of sinc^2(100t)')
plt.grid()
plt.show()