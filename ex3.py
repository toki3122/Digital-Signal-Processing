import numpy as np
import matplotlib.pyplot as plt

fs = 8
Ts = 1 / fs
t = np.linspace(0, 2, 1000)
ts = np.arange(0, 2 + Ts, Ts)
x = (np.cos(2*np.pi*t) + np.cos(14*np.pi*t) + np.cos(18*np.pi*t))
xa = 3 * np.cos(2*np.pi*t)
xs = (np.cos(2*np.pi*ts) + np.cos(14*np.pi*ts) + np.cos(18*np.pi*ts))
plt.figure(figsize=(10,6))
plt.plot(t, x)
plt.plot(t, xa, '--')
plt.stem(ts, xs, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title('Aliasing Demonstration')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()