import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
system1_num = [1]
system1_den = [1, -1]
system2_num = [1, -1]
system2_den = [1]
cascade_num = np.convolve(system1_num,system2_num)
cascade_den = np.convolve(system1_den,system2_den)
omega, frequency_response = freqz(cascade_num,cascade_den,worN=1024)
plt.subplot(2, 1, 1)
plt.plot(omega,np.abs(frequency_response))
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.title("Magnitude Response of Cascaded System")
plt.subplot(2, 1, 2)
plt.plot(omega,np.angle(frequency_response))
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Phase (radians)")
plt.title("Phase Response of Cascaded System")
plt.grid()
plt.tight_layout()
plt.show()
