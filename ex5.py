import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * 10 * t)
b = 4                    
L = 2 ** b              
DR = np.max(x) - np.min(x)   
q = DR / L               
y = np.sign(x) * q * np.floor((np.abs(x) / q) + 0.5)
e = x - y

plt.figure(figsize=(10, 8))
plt.subplot(3,1,1)
plt.plot(t, x, 'b')
plt.title('Input Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3,1,2)
plt.plot(t, y, 'r')
plt.title('4-bit Uniform Mid-Tread Quantized Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3,1,3)
plt.plot(t, e, 'g')
plt.title('Quantization Error')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.grid()

plt.tight_layout()
plt.show()