import numpy as np
import matplotlib.pyplot as plt
fs = 5000
t0 = 0.1
t = np.arange(-t0, t0, 1/fs)
m1 = np.sinc(100*t)
m2 = np.sinc(100*t)**2
fc1 = 250
fc2 = 750
y = m1*np.cos(2*np.pi*fc1*t) + m2*np.cos(2*np.pi*fc2*t)
N = 4096
Y = np.fft.fft(y, N)
f = np.fft.fftfreq(N, 1/fs)
plt.plot(np.fft.fftshift(f),
 np.abs(np.fft.fftshift(Y)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum')
plt.grid()
plt.xlim(-1500, 1500)
plt.show() 