import numpy as np

def convolution_sum(x, nx, h, nh):
    kmin = np.min(nx) + np.min(nh)
    kmax = np.max(nx) + np.max(nh)
    k = np.arange(kmin, kmax + 1)
    y = np.convolve(x, h)
    return y, k

x  = np.array([1,2,3])
nx = np.arange(0,3)
h1  = np.array([1, 1])
nh1 = np.array([0, 1])
h2  = np.array([2, 1, 1])
nh2 = np.array([-1, 0, 1])

# Left side: (x * h1) * h2
y1, ny1 = convolution_sum(x, nx, h1, nh1)
y_left, n_left = convolution_sum(y1, ny1, h2, nh2)

# Right side: x * (h1 * h2)
y2, ny2 = convolution_sum(h1, nh1, h2, nh2)
y_right, n_right = convolution_sum(x, nx, y2, ny2)

print("Left side  y :", y_left)
print("Left side  n :", n_left)
print("Right side y :", y_right)
print("Right side n :", n_right)
print()
print("Sequences equal?", np.array_equal(y_left, y_right))
print("Support equal?  ", np.array_equal(n_left, n_right))