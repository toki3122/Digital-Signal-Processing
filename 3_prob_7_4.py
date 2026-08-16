import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10, 11)
x = np.exp(1j * np.pi / 8 * n)
x_conjugate = np.conjugate(x)
y = x * x_conjugate
imag_y = np.where(np.abs(y.imag) < 1e-10, 0, y.imag)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.stem(n, np.real(y))
plt.xlabel('n')
plt.ylabel('Real')
plt.title('Real Part of y[n]')
plt.grid()

plt.subplot(2, 1, 2)
plt.stem(n, imag_y)
plt.xlabel('n')
plt.ylabel('Imaginary')
plt.title('Imaginary Part of y[n]')
plt.grid()

plt.tight_layout()
plt.show()