import numpy as np
import matplotlib.pyplot as plt
n = np.arange(-5, 11)
delta_n = np.zeros(len(n))
delta_n[n == 0] = 1
delta_n_minus_1 = np.zeros(len(n))
delta_n_minus_1[n == 1] = 1
h = delta_n - delta_n_minus_1
step_response = np.cumsum(h)
plt.subplot(2, 1, 1)
plt.stem(n, h)
plt.xlabel("n")
plt.ylabel("h[n]")
plt.title("Impulse Response")
plt.grid()
plt.subplot(2, 1, 2)
plt.stem(n, step_response)
plt.xlabel("n")
plt.ylabel("Step Response")
plt.title("Step Response")
plt.grid()
plt.tight_layout()
plt.show()
