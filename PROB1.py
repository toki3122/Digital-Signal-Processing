import numpy as np
import matplotlib.pyplot as plt

Fo = 75              
To = 1 / Fo          
Fs_values = [4 * Fo, 2 * Fo, Fo]

plt.figure()
for i in range(3):
    Ts = 1 / Fs_values[i]
    t = np.linspace(0, 3 * To, 1000)
    xc = (10 * np.cos(120 * np.pi * t) +
          5 * np.sin(100 * np.pi * t + np.pi / 6) +
          4 * np.sin(150 * np.pi * t + np.pi / 4))
    
    t1 = np.arange(0, 3 * To + Ts, Ts)
    xs = (10 * np.cos(120 * np.pi * t1) +
          5 * np.sin(100 * np.pi * t1 + np.pi / 6) +
          4 * np.sin(150 * np.pi * t1 + np.pi / 4))

    plt.subplot(3, 1, i+1)
    plt.plot(t, xc, color='b', label='Continuous Signal')
    plt.stem(t1, xs, linefmt='r-', markerfmt='ro', basefmt='k-')
    plt.title(f'Continuous and Sampled Signals (Fs = {Fs_values[i]} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

plt.tight_layout()
plt.show()