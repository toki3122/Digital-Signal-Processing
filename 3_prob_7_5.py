import numpy as np
import matplotlib.pyplot as plt

def unit_impulse(n0, n1, n2):
    n = np.arange(n1, n2 + 1, 1)
    x = (n == n0)
    return x, n

n1 = 0
n2 = 7
d0, n = unit_impulse(0, n1, n2)
d1, n = unit_impulse(1, n1, n2)
d2, n = unit_impulse(2, n1, n2)
d3, n = unit_impulse(3, n1, n2)
d4, n = unit_impulse(4, n1, n2)
d5, n = unit_impulse(5, n1, n2)
d6, n = unit_impulse(6, n1, n2)
d7, n = unit_impulse(7, n1, n2)
x1 = d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
x2 = d0 +(-1) * d1 + d2 +(-1) * d3 + d4 +(-1) * d5 + d6 +(-1) * d7
x3 = d0 + d1 +(-1) * d2 +(-1) * d3 + d4 + d5 +(-1) * d6 +(-1) * d7
x4 = d0 + d1 + d2 + d3 +(-1) * d4 +(-1) * d5 +(-1) * d6 +(-1) * d7

E1 = np.sum(np.abs(x1)**2)
E2 = np.sum(np.abs(x2)**2)
E3 = np.sum(np.abs(x3)**2)
E4 = np.sum(np.abs(x4)**2)

print("Energy of x1[n] =", E1)
print("Energy of x2[n] =", E2)
print("Energy of x3[n] =", E3)
print("Energy of x4[n] =", E4)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.stem(n, x1)
plt.xlabel('n')
plt.ylabel('x1[n]')
plt.title('x1[n]')
plt.grid()

plt.subplot(2, 2, 2)
plt.stem(n, x2)
plt.xlabel('n')
plt.ylabel('x2[n]')
plt.title('x2[n]')
plt.grid()

plt.subplot(2, 2, 3)
plt.stem(n, x3)
plt.xlabel('n')
plt.ylabel('x3[n]')
plt.title('x3[n]')
plt.grid()

plt.subplot(2, 2, 4)
plt.stem(n, x4)
plt.xlabel('n')
plt.ylabel('x4[n]')
plt.title('x4[n]')
plt.grid()

plt.tight_layout()
plt.show()