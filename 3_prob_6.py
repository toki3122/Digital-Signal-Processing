import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 1, 1])
y = np.array([3, 5, 8, 13, 21])
nx = np.arange(0, 4)
ny = np.arange(-1, 4)

rxy = np.correlate(x, y, mode='full')
lag_xy = np.arange(-len(y) + 1, len(x))
ryx = np.correlate(y, x, mode='full')
lag_yx = np.arange(-len(x) + 1, len(y))

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.stem(lag_xy, rxy)
plt.xlabel('Lag')
plt.ylabel('Cross correlation')
plt.title(r'$r_{xy}[k]$')
plt.grid()

plt.subplot(2, 1, 2)
plt.stem(lag_yx, ryx)
plt.xlabel('Lag')
plt.ylabel('Cross correlation')
plt.title(r'$r_{yx}[k]$')
plt.grid()

plt.tight_layout()
plt.show()