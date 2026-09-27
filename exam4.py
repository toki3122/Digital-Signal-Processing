import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ct
fs=8000
nyq=fs/2
wp=[1500/nyq,2500/nyq]
ws=[1000/nyq,3000/nyq]
Ap=3
As=40
N,Wn=signal.cheb2ord(wp=wp,ws=ws,gpass=Ap,gstop=As,analog=False)
b,a=signal.cheby2(N=N,Wn=Wn,btype='bandstop',rs=As,analog=False,output='ba')
w,h=signal.freqz(b,a,fs=fs)
plt.plot(w/1000,20*np.log10(np.abs(h)))
plt.grid()
plt.show()
