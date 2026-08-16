import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * 5 * t)      

b = [1, 2, 4]                      

DR = np.max(x) - np.min(x)         

for i in range(len(b)):
    L = 2 ** b[i]                  
    q = DR / L                     
    y = np.sign(x) * q * np.floor((abs(x) / q) + 0.5)
    plt.subplot(3, 1, i + 1)
    plt.plot(t, x, 'b')
    plt.plot(t, y, 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Quantization with {b[i]} bits')
    plt.grid()
plt.tight_layout()
plt.show()