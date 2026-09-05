import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
numerator = [1/3, 1/3, 1/3]
denominator = [1]
frequency, response = freqz(numerator,denominator,worN=1024)
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(frequency,np.abs(response))
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.title("Magnitude Response")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(frequency,np.unwrap(np.angle(response)))
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Phase (radians)")
plt.title("Phase Response")
plt.grid()
plt.tight_layout()
plt.show()