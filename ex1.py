import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
xc = np.sin(20*np.pi*t) + np.sin(50*np.pi*t)
Fs = [100, 25]
plt.figure(figsize=(10,8))
for i in range(2):
    fs = Fs[i]
    Ts = 1/fs
    ts = np.arange(0, 1+Ts, Ts)
    xs = np.sin(20*np.pi*ts) + np.sin(50*np.pi*ts)
    plt.subplot(2,1,i+1)
    plt.plot(t, xc, 'b', label='Original Signal')
    plt.stem(ts, xs, linefmt='r-', markerfmt='ro', basefmt='k-', label='Samples')
    plt.title(f'Sampling at Fs = {fs} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
plt.tight_layout()
plt.show()