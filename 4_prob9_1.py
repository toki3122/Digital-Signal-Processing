import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
z = sp.symbols('z')
X1 = z**(-1)
X2 = 1
X3 = z**(-2) * (2 - z**(-1)) / (1 - z**(-1))**2
print("Z-transform of x1[n] =", X1)
print("Z-transform of x2[n] =", X2)
print("Z-transform of x3[n] =", sp.simplify(X3))