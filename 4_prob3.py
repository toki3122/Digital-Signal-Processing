import control as ct
import numpy as np
import matplotlib.pyplot as plt

num = [1, -2, 2, -1]
den = [1, -1.7, 0.8, -0.1]
sys = ct.tf(num, den, dt=True)
poles = ct.poles(sys)
print("Zeros:")
print(ct.zeros(sys))
print("\nPoles:")
print(poles)
if np.all(np.abs(poles)<1):
    print("System is STABLE")
else:
    print("System is UNSTABLE")
ct.pzmap(sys, plot=True)
plt.show()