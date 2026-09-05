import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import signal

H1 = ct.tf([1, 0.5], [1], dt=True)
H2 = ct.tf([1], [1, -0.5], dt=True)
H3 = ct.tf([1, 0.5], [1, -0.5], dt=True)
systems = [
    (H1, "H1(z)"),
    (H2, "H2(z)"),
    (H3, "H3(z)")
]
angle = np.linspace(0, 2 * np.pi, 500)
plt.figure(figsize=(15, 5))
for i, (sys, title) in enumerate(systems):
    print(title)
    print("Zeros:")
    print(ct.zeros(sys))
    print("\nPoles:")
    print(ct.poles(sys))
    print("\n" + "-" * 30)
    plt.subplot(1, 3, i + 1)
    ct.pzmap(sys)
    plt.plot(
        np.cos(angle),
        np.sin(angle),
        'k--',
    )
    plt.title(title)
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid()
    plt.axis("equal")
plt.tight_layout()
plt.show()