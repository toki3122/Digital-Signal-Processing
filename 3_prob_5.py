import numpy as np
import matplotlib.pyplot as plt

def signal_shift(x, n1, no):
    return x, n1 + no

N = 200
n = np.arange(1,N+1)
x = np.cos(0.2 * np.pi * n) + 0.5 * np.cos(0.6 * np.pi * n)
x_shift, n_shift = signal_shift(x, n, 20)
x_delayed = np.zeros(N)
for i in range(N):
    if n_shift[i] < N:
        x_delayed[n_shift[i]] = x_shift[i]

y = x + 0.1 * x_delayed
ryy = np.correlate(y, y, mode='full')
rxy = np.correlate(x, y, mode='full')
lag = np.arange(-N + 1, N)
# peak = np.argmax(rxy)
# delay = lag[peak]
td=np.argmax(ryy)-np.argmax(rxy)
print("Estimated delay =",td,"samples")
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(n, x)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Original Signal x[n]')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(n, y)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Received Signal y[n]')
plt.grid()


plt.subplot(2, 2, 3)
plt.plot(lag, ryy)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of y[n]')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(lag, rxy)
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation of x[n] and y[n]')
plt.grid()

plt.tight_layout()
plt.show()