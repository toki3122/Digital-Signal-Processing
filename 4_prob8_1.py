import numpy as np
import matplotlib.pyplot as plt
n = np.arange(0, 21)
h = np.zeros(len(n))
h[0] = 1
for k in range(1, len(n)):
 h[k] = 0.5 * h[k - 1]
plt.stem(n, h)
plt.xlabel("n")
plt.ylabel("h[n]")
plt.title("Impulse Response of the Discrete-Time System")
plt.grid()
plt.show()
if np.sum(np.abs(h)) < np.inf:
 print("The system is stable.")
else:
 print("The system is unstable.")