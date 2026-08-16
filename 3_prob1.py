import numpy as np
import matplotlib.pyplot as plt

def signal_fold(x, n):
    return np.flip(x), -np.flip(n)

def signal_add(x1, n1, x2, n2):
    min_n = min(np.min(n1),np.min(n2))
    max_n = max(np.max(n1),np.max(n2))
    n = np.arange(min_n, max_n + 1)
    y1 = np.zeros(len(n))
    y2 = np.zeros(len(n))
    y1[np.isin(n,n1)]=x1
    y2[np.isin(n,n2)]=x2    
    return y1 + y2,n

n = np.arange(-20, 21)          
x = np.sin(n) + np.cos(n)
x_fold, n_fold = signal_fold(x, n)
x_even, n_even = signal_add(x, n, x_fold, n_fold)
x_even = x_even / 2
x_odd, n_odd = signal_add(x, n, -x_fold, n_fold)
x_odd = x_odd / 2

plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.stem(n, x, basefmt=" ")
plt.title(r"Original discrete signal  $x[n]=\sin n + \cos n$")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(n_even, x_even, basefmt=" ")
plt.title(r"Even component  $x_e[n]=\cos n$")
plt.xlabel("n")
plt.ylabel(r"$x_e[n]$")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(n_odd, x_odd, basefmt=" ")
plt.title(r"Odd component  $x_o[n]=\sin n$")
plt.xlabel("n")
plt.ylabel(r"$x_o[n]$")
plt.grid(True)

plt.tight_layout()
plt.show()