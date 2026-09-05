import numpy as np
import matplotlib.pyplot as plt
n = np.arange(-10, 11)
x1 = np.sin(0.3 * np.pi * n)
x2 = np.cos(0.2 * np.pi * n)
a = 2
b = 3
combined_input = a * x1 + b * x2
def system_response(x, n):
   output = np.zeros(len(n))
   for i in range(len(n)):
      position = np.array([], dtype=int)
      if n[i] % 2 == 0:
         position = np.where(n == n[i] // 2)[0]
      if len(position) > 0:
         output[i] = x[position[0]]
   return output
y_combined = system_response(combined_input, n)
y1 = system_response(x1, n)
y2 = system_response(x2, n)
linear_combination = a * y1 + b * y2
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.stem(n, y_combined, label="T[a*x1 + b*x2]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("T[a*x1 + b*x2]")
plt.subplot(2, 1, 2)
plt.stem(n, linear_combination, markerfmt='x', label="a*T[x1] + b*T[x2]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("a*T[x1] + b*T[x2]")
plt.grid()
plt.tight_layout()
plt.show() 
