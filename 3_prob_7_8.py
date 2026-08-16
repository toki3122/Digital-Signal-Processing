import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10, 11)
e1 = np.cos(0.3 * np.pi * n)
e2 = np.cos(0.6 * np.pi * n)
o1 = np.sin(0.3 * np.pi * n)
o2 = np.sin(0.6 * np.pi * n)
even_even = e1 * e2
odd_odd = o1 * o2
even_odd = e1 * o1

plt.figure(figsize=(10, 7))

plt.subplot(3,1, 1)
plt.stem(n, even_even)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Even × Even')
plt.grid()

plt.subplot(3,1, 2)
plt.stem(n, odd_odd)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Odd × Odd')
plt.grid()

plt.subplot(3,1, 3)
plt.stem(n, even_odd)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Even × Odd')
plt.grid()

plt.tight_layout()
plt.show()