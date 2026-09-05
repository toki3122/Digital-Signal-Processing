import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 1, 2, 3, 5, 8, 13])
N = 2*len(x) - 1
X = np.fft.fft(x, N)
r = np.fft.ifft(np.abs(X)**2)
r = np.real(r)
r = np.fft.fftshift(r)
lags = np.arange(-(len(x)-1), len(x))
plt.stem(lags, r)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of x[n]')
plt.grid()
plt.show()
print(r) 