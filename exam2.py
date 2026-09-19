import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ct

# Design a notch filter using pole-zero placement method with following specifications:
# - notch frequency 50Hz
# - 3dB width of the notch +5 Hz
# - sampling frequency 500H z
# Showthe magnitude and phase response of the filter. Consider a signal, (t) = 2 cos (60πt)+cos (100πt).
# If x[n] is the signal obtained from x(t) by sampling it at a frequency 500H z, show the output signal
# after x[n] is passed through your designed notch filter.

fs=500
fp=50
bw=10
r=1-(bw/fs)*np.pi
wo=2*np.pi*(fp/fs)
z1=np.exp(1j*wo)
z2=np.exp(-1j*wo)
p1=r*np.exp(1j*wo)
p2=r*np.exp(-1j*wo)
zeros=[z1,z2]
poles=[p1,p2]
num=np.poly(zeros).real
den=np.poly(poles).real
sys=ct.tf(num,den,dt=True)
w,h=signal.freqz(num,den,fs=fs)
#2 cos (60πt)+cos (100πt)
To = 1/30
ts = 1/fs
t1 = np.arange(0, 4*To + ts, ts)
xs = 2*np.cos(60*np.pi*t1) + np.cos(100*np.pi*t1)
y = signal.lfilter(num, den, xs)
plt.subplot(3,1,1)
plt.plot(w,20*np.log10(np.abs(h)))
plt.grid()
plt.subplot(3,1,2)
plt.plot(w,np.angle(h,deg=True))
plt.grid()
plt.subplot(3,1,3)
ct.pzmap(sys)
plt.tight_layout()
plt.show()
plt.subplot(2,1,1)
plt.plot(t1,xs)
plt.subplot(2,1,2)
plt.plot(t1,y)
plt.tight_layout()
plt.show()

# Design a bandstop filter with the following property using pole-zero placement method.
# - Center frequency, omegao =pi/10
# - Bandstopwidth, bw = pi/20
# Show the frequency response of the filter and mention the filter coefficients.

import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import signal
fs=500
w0=np.pi/10
bw=np.pi/20
r=1-(bw/fs)*np.pi
z1=np.exp(1j*w0)
z2=np.exp(-1j*w0)
p1=r*np.exp(1j*w0)
p2=r*np.exp(-1j*w0)
zeros=[z1,z2]
poles=[p1,p2]
print(zeros)
print(poles)
num=np.poly(zeros).real
den=np.poly(poles).real
print(num)
print(den)
w,h=signal.freqz(num,den,fs=fs)
sys=ct.tf(num,den,dt=True)
plt.subplot(3,1,1)
plt.plot(w,20*np.log10(np.abs(h)))
plt.subplot(3,1,2)
plt.plot(w,np.angle(h,deg=True))
plt.subplot(3,1,3)
ct.pzmap(sys)
plt.tight_layout()
plt.show()

# A discrete bandpass filter with following specifications is to be designed using bilinear transformation
# from a low pass filter prototype.
# - passband, 200 – 300Hz
# - sampling frequency, 2KH z
# - filter order, N =2
# The transfer function of prototype low pass filter is H(s) =1/s+1
# Determine the filter coefficients and show the frequency response of the filter.

fs=2000
fl=200
fu=300

wl=2*fs*np.tan(np.pi*fl/fs)   # prewarped lower edge (rad/s)
wu=2*fs*np.tan(np.pi*fu/fs)   # prewarped upper edge (rad/s)
w0=np.sqrt(wl*wu)             # analog center frequency
bw=wu-wl                      # analog bandwidth (rad/s)

num=[1]
den=[1,1]
num_bpfa,den_bpfa=signal.lp2bp(num,den,w0,bw)
num_bpf,den_bpf=signal.bilinear(num_bpfa,den_bpfa,fs)
w,h=signal.freqz(num_bpf,den_bpf,worN=256,fs=fs)
plt.subplot(2,1,1)
plt.plot(w,h)
plt.grid()
plt.subplot(2,1,2)
plt.plot(w,np.angle(h,deg=True))
plt.grid()
plt.tight_layout()
plt.show()

# Write a Python code to design a Butterworth digital bandpass filter for the following specifications:
# - Passband frequencies are 1500H z and 2500Hz.
# - Stopband frequencies are 1000H z and 3000H z.
# - Sampling frequency Fs = 8kHz.
# - Passband ripple is 1dB and stopband attenuation is 30 dB.
# - Use BLT method for transformation.

fs=8000
fp1=1500
fp2=2500
fs1=1000
fs2=3000
Ap=1
As=30
wp1=(2*fs)*np.tan(np.pi*fp1/fs)
wp2=(2*fs)*np.tan(np.pi*fp2/fs)
ws1=(2*fs)*np.tan(np.pi*fs1/fs)
ws2=(2*fs)*np.tan(np.pi*fs2/fs)
wp=[wp1,wp2]
ws=[ws1,ws2]
N,Wn=signal.buttord(wp,ws,Ap,As,analog=True)
wn=np.arctan(Wn/(2*fs))*(fs/np.pi)
b,a=signal.butter(N,wn,btype='bandpass',analog=False,output='ba',fs=fs)
w,h=signal.freqz(b,a,fs=fs)
plt.plot(w/1000,20*np.log10(np.abs(h)))
plt.axvline(fp1/1000,linestyle='--')
plt.axvline(fp2/1000,linestyle='--')
plt.grid()
plt.show()

# Design of a Chebyshev Type I digital highpass filter for the following specifications:
# - Passband frequency is 2500H z.
# - Stopband frequency is 1500H z.
# - Sampling frequency Fs = 8k Hz.
# - Passband ripple is 3dB and stop attenuation is 40dB.
# - Use BLT method for transformation.

fs=8000
fpass=2500
fstop=1500
Ap=3
As=40
wp=2*fpass/fs
ws=2*fstop/fs
N,wn=signal.cheb1ord(ws=ws,wp=wp,gpass=Ap,gstop=As,analog=False)
b,a=signal.cheby1(N=N,Wn=wn,analog=False,rp=Ap,btype='highpass',output='ba')
w,h=signal.freqz(b,a,fs=fs)
plt.plot(w/1000,20*np.log10(np.abs(h)))
plt.axvline(fpass/1000,linestyle='--')
plt.axvline(fstop/1000,linestyle='--')
plt.grid()
plt.show()

# Design a Chebyshev Type II digital bandpass filter for the following specifications:
# - Passband frequencies are from 1500 — 2500H z with an attenuation of 3dB.
# - Stopband frequencies are below 1000H z and above 3000H z with the passband ripple of 40dB.
# - Sampling frequency Fs = 8kHz.
# - Use BLT method for transformation.
fs=8000
nyq=fs/2.0
wp=[1500/nyq,2500/nyq]
ws=[1000/nyq,3000/nyq]
As=40
Ap=3
n,wn=signal.cheb2ord(wp,ws,gpass=Ap,gstop=As,analog=False)
b,a=signal.cheby2(N=n,Wn=wn,rs=As,analog=False,btype='bandpass',output='ba')
w,h=signal.freqz(b,a,fs=fs)
plt.plot(w/1000,20*np.log(np.abs(h)))
plt.axvline(1.5,linestyle='--')
plt.axvline(2.5,linestyle='--')
plt.grid()
plt.show()