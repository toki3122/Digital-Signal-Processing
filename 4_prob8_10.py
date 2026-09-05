import numpy as np
import matplotlib.pyplot as plt
fs = 100
input_frequency = 5
number_of_samples = 200
time = np.arange(number_of_samples) / fs
input_signal = np.sin(2 * np.pi *input_frequency *time)
output_1 = input_signal[::2]
reversed_output = output_1[::-1]
minimum_length = min(len(output_1),len(reversed_output))
output_signal = 0.5 * (output_1[:minimum_length]+reversed_output[:minimum_length])
plt.subplot(2, 1, 1)
plt.plot(time,input_signal,label="Input Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Input Signal: 5 Hz Sine Wave")
output_time = np.arange(minimum_length) / fs
plt.subplot(2, 1, 2)
plt.plot(
 output_time,
 output_signal,
 label="Output Signal"
)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Output Signal of Cascaded System")
plt.grid()
plt.tight_layout()
plt.show()