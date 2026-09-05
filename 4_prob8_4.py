import numpy as np
import matplotlib.pyplot as plt
n = np.arange(-10, 11)
a_values = [0.25, 0.5, 1.0, 2.0]
plt.figure(figsize=(10, 8))
for index, a in enumerate(a_values):
 impulse_response = a ** n
 plt.subplot(2, 2, index + 1)
 plt.stem(n, impulse_response)
 plt.xlabel("n")
 plt.ylabel("h[n]")
 plt.title(f"Impulse Response for a = {a}")
 plt.grid()
plt.tight_layout()
plt.show() 