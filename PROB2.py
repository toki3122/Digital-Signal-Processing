import numpy as np
import matplotlib.pyplot as plt

Fo = 125            
Fs = 800            
Ts = 1 / Fs
t = np.linspace(-10*Ts, 10*Ts, 1000)
xc = (10*np.cos(250*np.pi*t + np.pi/3) + 5*np.sin(200*np.pi*t + np.pi*(75/180)))
n = np.arange(-10, 11)
t1 = n * Ts
xs = (10*np.cos(250*np.pi*t1 + np.deg2rad(60)) + 5*np.sin(200*np.pi*t1 + np.deg2rad(75)))

plt.figure(figsize=(10,10))

plt.subplot(4,1,1)
plt.plot(t, xc, 'b')
plt.title("Original Continuous Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,1,2)
plt.plot(t, xc, 'b')
plt.stem(t1, xs, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title("Sampled Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,1,3)
plt.stem(n, xs, linefmt='g-', markerfmt='go', basefmt='k-')
plt.title("Discrete Signal x[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,1,4)
plt.plot(t1, xs, 'm-o')
plt.title("Final Output")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()