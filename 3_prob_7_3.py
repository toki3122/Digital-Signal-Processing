import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10, 11)
x = np.exp(1j * np.pi / 4 * n)
x_real = np.real(x)
x_imag = np.imag(x)
x_reconstructed = x_real + 1j * x_imag

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.stem(n, x_real)
plt.xlabel('n')
plt.ylabel('Real{x[n]}')
plt.title('Real Part of x[n]')
plt.grid()

plt.subplot(3, 1, 2)
plt.stem(n, x_imag)
plt.xlabel('n')
plt.ylabel('Imag{x[n]}')
plt.title('Imaginary Part of x[n]')
plt.grid()

plt.subplot(3, 1, 3)
plt.stem(n, np.real(x), label='Original')
plt.stem(n, np.real(x_reconstructed), markerfmt='rx',
         linefmt='r--', label='Reconstructed')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Original vs Reconstructed')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()