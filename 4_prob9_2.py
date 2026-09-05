import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
z, n = sp.symbols('z n', integer=True)
X1 = z**(-2)
print("Inverse Z-transform of X1(z) = delta[n-2]")
X2 = 1 / (1 - z**(-1))**2
print("Inverse Z-transform of X2(z) = (n+1)u[n]")
n = np.arange(0, 10)
x1 = (n == 2).astype(int)
x2 = n + 1
plt.subplot(2, 1, 1)
plt.stem(n, x1)
plt.xlabel("n")
plt.ylabel("x1[n]")
plt.title("x1[n] = delta[n-2]")
plt.subplot(2, 1, 2)
plt.stem(n, x2)
plt.xlabel("n")
plt.ylabel("x2[n]")
plt.title("x2[n] = (n+1)u[n]")
plt.grid()
plt.tight_layout()
plt.show()