import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def unit_impulse(no, n1, n2):
    n = np.arange(n1, n2 + 1, 1)
    x = (n == no)
    return x, n

n1 = -20
n2 = 25

# System:
# y[n] - 4y[n-1] + 4y[n-2] = x[n] - x[n-1]

coeff_x = [1, -1]
coeff_y = [1, -4, 4]
x, n = unit_impulse(0, n1, n2)
x = x.astype(float)
h1 = np.zeros(len(n))
for i in range(len(n)):
    if i >= 1:
        h1[i] = 4*h1[i-1]
    if i >= 2:
        h1[i] = h1[i] - 4*h1[i-2]
    if i >= 1:
        h1[i] = h1[i] - x[i-1]
    h1[i] = h1[i] + x[i]

h2 = signal.lfilter(coeff_x, coeff_y, x)
plt.subplot(3, 1, 1)
plt.stem(n, h1)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Impulse Response by Difference Equation')
plt.subplot(3, 1, 2)
plt.stem(n, h2)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Impulse Response using lfilter')

u = (n >= 0).astype(float)
x = (5 + 3*np.cos(0.2*np.pi*n) + 4*np.sin(0.6*np.pi*n))*u
y = signal.lfilter(coeff_x, coeff_y, x)
plt.subplot(3,1,3)
plt.stem(n, y)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Output of the System')
plt.grid()
plt.tight_layout()
plt.show()