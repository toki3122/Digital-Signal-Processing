import numpy as np
from scipy.signal import ss2tf
A = np.array([
 [0, 1],
 [-1/6, -5/6]
])
B = np.array([
 [0],
 [1]
])
C = np.array([
 [1, 0]
])
D = np.array([
 [0]
])
numerator, denominator = ss2tf(A, B, C, D)
print("Numerator Coefficients:")
print(numerator)
print("\nDenominator Coefficients:")
print(denominator)