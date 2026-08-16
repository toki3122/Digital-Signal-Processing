import numpy as np
import matplotlib.pyplot as plt

f = 1300       
fs = 8000 
t = np.arange(0, 2, 1/fs)
x = np.sin(2*np.pi*f*t)
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t[:200], x[:200])
plt.title("Original Signal (1300 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

x_down = x[::2]
fs_down = fs // 2
t_down = np.arange(len(x_down)) / fs_down
plt.subplot(2,1,2)
plt.plot(t_down[:100], x_down[:100])
plt.title("Downsampled Signal by factor 2")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()