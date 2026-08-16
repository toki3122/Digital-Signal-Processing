import numpy as np
import matplotlib.pyplot as plt

def unit_step(n0, n1, n2):
    n = np.arange(n1, n2 + 1)
    x = (n >= n0)
    return x, n

def unit_impulse(n0, n1, n2):
    n = np.arange(n1, n2 + 1)
    x = (n == n0)
    return x, n

def unit_ramp(no, n1, n2, n3, n4):
    n = np.arange(n3, n4 + 1)
    x = (n - no) * ((n >= no) & (n <= n2))
    return x, n

u1,n= unit_step(0, -10, 10)
u2,n= unit_step(5, -10, 10)
x1=u1+(-1)*u2
plt.figure(figsize=(10, 7))
plt.subplot(2, 2, 1)
plt.stem(n, x1)
plt.title(r'$x_1[n] = u[n] - u[n-5]$')
plt.xlabel('n')
plt.ylabel('x1[n]')
plt.grid()

x1,n=unit_impulse(0,-10,10)
plt.subplot(2, 2, 2)
plt.stem(n, x1)
plt.title(r'$x_1[n] = delta[n]$')
plt.xlabel('n')
plt.ylabel('x1[n]')
plt.grid()

u1,n= unit_step(-5, -10, 10)
u2,n= unit_step(5, -10, 10)
x1=u1+(-1)*u2
plt.subplot(2, 2, 3)
plt.stem(n, x1)
plt.title(r'$x_1[n] = u[n+5] - u[n-5]$')
plt.xlabel('n')
plt.ylabel('x1[n]')
plt.grid()

x1,n=unit_ramp(0,0,5,-10,10)
plt.subplot(2, 2, 4)
plt.stem(n, x1)
plt.title(r'$x_1[n] = r[n]-u[n-6]-r[n-6]$')
plt.xlabel('n')
plt.ylabel('x1[n]')
plt.grid()

plt.tight_layout()
plt.show()