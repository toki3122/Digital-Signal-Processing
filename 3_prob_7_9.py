import numpy as np
import matplotlib.pyplot as plt

N = 200
n = np.arange(1, N + 1)
x = (np.sin(0.2 * np.pi * n) + 0.5 * np.sin(0.4 * np.pi * n))
rxx = np.correlate(x, x, mode='full')
lag = np.arange(-N + 1, N)
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(n, x)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Speech-like Signal')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(lag, rxx)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Speech-like Signal')
plt.grid()

plt.tight_layout()
plt.show()