import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
b = [0, 1, 0.5]
a = [1, -3/5, 2/25]
w, H = freqz(b, a)
plt.plot(w, np.abs(H))
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('|H(e^jw)|')
plt.title('Magnitude Frequency Response')
plt.grid()
plt.show()
