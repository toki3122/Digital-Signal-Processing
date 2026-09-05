import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
a0 = 0.8
b = [-a0, 1]
a = [1, -a0]
w, h = signal.freqz(b, a, worN=1024)
magnitude = 20*np.log10(np.maximum(abs(h), 1e-10))
phase = np.unwrap(np.angle(h))
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(w / np.pi, magnitude)
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.title('First-Order All-Pass Filter')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(w / np.pi, phase)
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Phase (rad)')
plt.grid()
plt.tight_layout()
plt.show() 