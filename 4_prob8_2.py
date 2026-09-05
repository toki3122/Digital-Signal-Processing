import numpy as np
from scipy.signal import tf2ss
num = [1, 0, 0]
den = [1, 4, -2]
A, B, C, D = tf2ss(num, den)
print("State Matrix A:")
print(A)
print("\nInput Matrix B:")
print(B)
print("\nOutput Matrix C:")
print(C)
print("\nFeedthrough Matrix D:")
print(D)
