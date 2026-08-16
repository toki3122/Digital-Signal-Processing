import numpy as np
import matplotlib.pyplot as plt

def signal_shift(x, n1, no):
    return x, n1 + no

def signal_fold(x, n):
    return np.flip(x), -np.flip(n)

x = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
n = np.arange(-4, 6)
x_fold, n_fold = signal_fold(x, n)
x1, n1 = signal_shift(x_fold, n_fold, -3)
x2, n2 = signal_shift(x_fold, n_fold, 4)

plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
plt.stem(n, x)
plt.title(r'$x[n]$')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(n1, x1)
plt.title(r'$x[-n-3]$')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(n2, x2)
plt.title(r'$x[-n+4]$')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()