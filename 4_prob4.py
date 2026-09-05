import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import signal

num = [2, 16, 44, 56, 32]
den = [3, 3, -15, 18, -12]

sys = ct.tf(num, den, dt=True)
print("Zeros:")
print(ct.zeros(sys))

print("\nPoles:")
print(ct.poles(sys))
ct.pzmap(sys)
plt.show()
w, h = signal.freqz(num, den, worN=1024)

plt.subplot(2, 1, 1)
plt.plot(w, np.abs(h))
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.title("Magnitude Frequency Response")
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(h))
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Phase (radians)")
plt.title("Phase Frequency Response")
plt.grid()
plt.tight_layout()
plt.show()