import numpy as np
import matplotlib.pyplot as plt
def unit_impulse(n0, n1, n2):
    n = np.arange(n1, n2 + 1, 1)
    x = (n == n0)
    return x, n

d0, n = unit_impulse(0, -5, 5)
d1, n = unit_impulse(-1, -5, 5)
d2, n = unit_impulse(1, -5, 5)
d3, n = unit_impulse(2, -5, 5)
x1 = d1 + d2
x2 = d1 + (-1)*d2
x3 = d0 + 2*d2 + d3
x4 = d0 + (-1)*d2 + d3

plt.figure(figsize=(10, 7))
plt.subplot(2, 2, 1)
plt.stem(n, x1)
plt.title(r'$x_1[n] = \delta[n+1] + \delta[n-1]$')
plt.xlabel('n')
plt.ylabel('x1[n]')
plt.grid()

plt.subplot(2, 2, 2)
plt.stem(n, x2)
plt.title(r'$x_2[n] = \delta[n+1] - \delta[n-1]$')
plt.xlabel('n')
plt.ylabel('x2[n]')
plt.grid()

plt.subplot(2, 2, 3)
plt.stem(n, x3)
plt.title(r'$x_3[n] = \delta[n] + 2\delta[n-1] + \delta[n-2]$')
plt.xlabel('n')
plt.ylabel('x3[n]')
plt.grid()

plt.subplot(2, 2, 4)
plt.stem(n, x4)
plt.title(r'$x_4[n] = \delta[n] - \delta[n-1] + \delta[n-2]$')
plt.xlabel('n')
plt.ylabel('x4[n]')
plt.grid()

plt.tight_layout()
plt.show()