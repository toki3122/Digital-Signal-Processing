import numpy as np
import matplotlib.pyplot as plt

def signal_fold(x, n):
    return np.flip(x), -np.flip(n)

n = np.arange(-10, 11)
x = np.zeros(len(n))
for i in range(len(n)):
    if abs(n[i]) <= 5:
        x[i] = 5 - abs(n[i])

x_fold, n_fold = signal_fold(x, n)
x_even = (x + x_fold) / 2
x_odd = (x - x_fold) / 2
x_reconstructed = x_even + x_odd

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Original Signal')
plt.grid()

plt.subplot(2, 2, 2)
plt.stem(n, x_even)
plt.xlabel('n')
plt.ylabel('xe[n]')
plt.title('Even Part')
plt.grid()

plt.subplot(2, 2, 3)
plt.stem(n, x_odd)
plt.xlabel('n')
plt.ylabel('xo[n]')
plt.title('Odd Part')
plt.grid()

plt.subplot(2, 2, 4)
plt.stem(n, x_reconstructed)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Reconstructed Signal')
plt.grid()

plt.tight_layout()
plt.show()