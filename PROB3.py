import numpy as np
import matplotlib.pyplot as plt
fo=15
f1=5*fo
f2=1.5*fo
t=np.linspace(0,2,1000)
t1=1/f1
t2=1/f2
ts1=np.arange(0,2+t1,t1)
ts2=np.arange(0,2+t2,t2)
x_t=.5*np.sin(14*np.pi*t)+(1/3)*np.sin(18*np.pi*t)+(1/5)*np.sin(24*np.pi*t)+(1/7)*np.sin(30*np.pi*t)
x_s1=.5*np.sin(14*np.pi*ts1)+(1/3)*np.sin(18*np.pi*ts1)+(1/5)*np.sin(24*np.pi*ts1)+(1/7)*np.sin(30*np.pi*ts1)
x_s2=.5*np.sin(14*np.pi*ts2)+(1/3)*np.sin(18*np.pi*ts2)+(1/5)*np.sin(24*np.pi*ts2)+(1/7)*np.sin(30*np.pi*ts2)
plt.subplot(2,1,1)
plt.plot(t, x_t)
plt.plot(ts1, x_s1,'r--')
plt.ylabel("Amplitude")
plt.title("Signal x(t)(at Fs=5Fo)")
plt.grid()
plt.subplot(2,1,2)
plt.plot(t, x_t)
plt.plot(ts2, x_s2,'r--')
plt.ylabel("Amplitude")
plt.title("Signal x(t)(at Fs=1.5Fo)")
plt.grid()
plt.tight_layout()
plt.show()