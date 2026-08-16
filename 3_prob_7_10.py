import numpy as np
import matplotlib.pyplot as plt

N = 200
n = np.arange(N)

x = 2*(np.sin(0.1 * np.pi * n) + 0.5 * np.sin(0.2 * np.pi * n)) #Male voice
y = (np.sin(0.2 * np.pi * n) + 0.5 * np.sin(0.4 * np.pi * n)) #Female voice
rxx = np.correlate(x, x, mode='full')
lag_x = np.arange(-N + 1, N)
ryy = np.correlate(y, y, mode='full')
lag_y = np.arange(-N + 1, N)
rxy = np.correlate(x, y, mode='full')
lag_xy = np.arange(-N + 1, N)
ryx = np.correlate(y, x, mode='full')
lag_yx = np.arange(-N + 1, N)
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(lag_x, rxx)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Male Voice')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(lag_y, ryy)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Female Voice')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(lag_xy, rxy)
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.title('Male → Female Cross-correlation')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(lag_yx, ryx)
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.title('Female → Male Cross-correlation')
plt.grid()

plt.tight_layout()
plt.show()