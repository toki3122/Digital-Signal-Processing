import numpy as np
import matplotlib.pyplot as plt

def unit_impulse(n0, n1, n2):
    n = np.arange(n1, n2 + 1)
    x = (n == n0).astype(int)
    return x, n

n1 = -5
n2 = 5
d2, n = unit_impulse(2, n1, n2)
d_minus2, n = unit_impulse(-2, n1, n2)
x1 = 2 * n * d2
x2 = n * (d_minus2 + d2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.stem(n, x1)
plt.xlabel('n')
plt.ylabel('x1[n]')
plt.title(r'$x_1[n] = 2n\delta[n-2]$')
plt.grid()

plt.subplot(1, 2, 2)
plt.stem(n, x2)
plt.xlabel('n')
plt.ylabel('x2[n]')
plt.title(r'$x_2[n] = n[\delta[n+2]+\delta[n-2]]$')
plt.grid()

plt.tight_layout()
plt.show()