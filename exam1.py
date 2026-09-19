from lcapy import n,delta,us,exp,cos,z
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import control as ct
###########################
#ZT AND IZT
###########################
x1=cos(n)
x2=exp(1j*n)
x3=0.5**n*us(n)
X1=x1.ZT()
X2=x2.ZT()
X3=x3.ZT()
print(X1)
print(X2)
print(X3)
print()
X=z**(-1)
x=X.IZT()
Y=1/(1-X)
y=Y.IZT()
print(x)
print(y)
print()

####################
#IZT +partial fraction expansion
###################

num=[1,2,-1]
den=[1,-1,0.3561]
r,p,k=signal.residuez(num,den)
print("Residue:",r)
print("Pole:",p)
print("Direct term:",k)
num_rc,den_rc=signal.invresz(r,p,k)
print("\nReconstructed Numerator:")
print (num_rc)
print("\nReconstructed Denominator:")
print(den_rc)

####################
#lccde
###################

n=np.arange(0,4)
x=np.ones(len(n))
c_x=[1]
c_y=[1,-0.5]
y=signal.lfilter(c_x,c_y,x)
print(y)
plt.stem(n,y)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.tight_layout()
plt.show()

#####################
#impulse response
#####################

def unit_impulse(n0,n1,n2):
    n=np.arange(n1,n2+1)
    x=(n==n0).astype(float)
    return x,n
x,n=unit_impulse(0,0,50)
c_x=[1]
c_y=[1,-1]
y=signal.lfilter(c_x,c_y,x)
print(y)
plt.stem(n,y)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.tight_layout()
plt.show()

######################
#pole-zero plot
######################

num=[1,2,-1]
den=[1,-1,0.3561]
sys=ct.tf(num,den,dt=True)
ct.pzmap(sys)
plt.show()

#######################
#frequency response
#######################

num= [1,-1.618,1]
den=[1,-1.5371,0.9025]
fs=512
n=256
w,h=signal.freqz(num,den,worN=n,fs=fs)
plt.subplot(2,1,1)
mag=20*np.log10(np.abs(h))
plt.plot(w,mag)
plt.title('Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.subplot(2,1,2)
phase=np.angle(h,deg=True)
plt.plot(w,phase)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.tight_layout()
plt.grid()
plt.show()

################################
#1st order IIR LPF filter design
################################

x=np.zeros(30)
x[0]=1 #unit impulse
num=[1]
den=[1,-1] # H(z) = 1 / (1 - z^-1)
fs=100
sys=ct.tf(num,den,dt=True)
h=signal.lfilter(num,den,x)# Obtain the impulse response
w,H=signal.freqz(num,den)
plt.subplot(2,2,1)
plt.stem(h)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.grid()
plt.subplot(2,2,2)
plt.plot((w/np.pi)*fs/2,20*np.log10(np.abs(H)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.subplot(2,2,3)
ct.pzmap(sys)
plt.grid()
plt.subplot(2,2,4)
plt.plot((w/np.pi)*fs/2,np.angle(H))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.tight_layout()
plt.show()

#############################
#MOVING AVERAGE FILTER DESIGN
#############################
#y[n]=1/3*{x[n]+x[n-1]+x[n-2]}
x=np.zeros(30)
x[0]=1 #unit impulse
num=[1/3,1/3,1/3]
den=[1]
h=signal.lfilter(num,den,x)# Obtain the impulse response
fs=100
w,H=signal.freqz(num,den)
sys=ct.tf(num,den,dt=True)
plt.subplot(2,2,1)
plt.stem(h)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.grid()
plt.subplot(2,2,2)
plt.plot((w/np.pi)*fs/2,20*np.log10(np.abs(H)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.subplot(2,2,3)
ct.pzmap(sys)
plt.grid()
plt.subplot(2,2,4)
plt.plot((w/np.pi)*fs/2,np.angle(H))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.tight_layout()
plt.show()

##############################
#digital resonator design
##############################

x=np.zeros(30)
x[0]=1 #unit impulse
r=0.9
fs=100
fc=10
w=2*np.pi*fc/fs
num=[1]
den=[1,-2*r*np.cos(w),r**2]
h=signal.lfilter(num,den,x)# Obtain the impulse response
w,H=signal.freqz(num,den)
sys=ct.tf(num,den,dt=True)
plt.subplot(2,2,1)
plt.stem(h)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.grid()
plt.subplot(2,2,2)
plt.plot((w/np.pi)*fs/2,20*np.log10(H))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.subplot(2,2,3)
ct.pzmap(sys)
plt.grid()
plt.subplot(2,2,4)
plt.plot((w/np.pi)*fs/2,np.angle(H))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.tight_layout()
plt.show()

###############################
#fir lpf
###############################

fs=8000.0
tw=500.0
pbe=1500.0
m=int(np.round(6*fs/tw))
filtlen=m+1
fc=pbe+tw/2.0
a=signal.firwin(numtaps=filtlen, cutoff=fc, fs=fs,window='blackman',pass_zero='lowpass')
n=512
f,H=signal.freqz(a,1,worN=n,fs=fs)
plt.subplot(2,1,1)
plt.plot(f,20*np.log10(np.abs(H)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.axvline(pbe,color='red',linestyle='--')
plt.axvline(pbe+tw,color='red',linestyle='--')
plt.grid()
plt.subplot(2,1,2)
plt.plot(f,np.degrees(np.unwrap(np.angle(H))))
plt.xlabel("freq")
plt.ylabel("angle(deg)")
plt.grid()
plt.tight_layout()
plt.show()

######################
#fir bpf window method
######################

fs=100
m=50
fl=10
fu=20
filtlen=m+1
fc=[fl,fu]
a=signal.firwin(numtaps=filtlen,cutoff=fc,window='hamming',fs=fs,pass_zero='bandpass')
n=512
f,H=signal.freqz(a,1,worN=n,fs=fs)
plt.subplot(2,1,1)
plt.plot(f, 20 * np.log10(np.abs(H)))
plt.xlabel("Frequency (Hz) ")
plt.ylabel("Magnitude (dB)")
plt.axvline(fl, color='red', linestyle='--', label=f'Lower cutoff ({fl} Hz)')
plt.axvline(fu, color='purple', linestyle='--',label=f'Upper cutoff ({fu} Hz)')
plt.axhline(-3, color='orange', linestyle='-.', label='-3 dB Mark')
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(f, np.degrees(np.unwrap(np.angle(H))), color='green')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Angle (Degree)")
plt.grid()
plt.tight_layout()
plt.show()

#####################################
#filter design using pole-zero method
#####################################

# Design a bandpass digital filter using pole-zero placement method to meet the following specifications:
# - complete signal rejection at dc and 250 Hz
# - a narrow passband centered at 125 Hz
# - a 3dB bandwidth of 10 Hz
# The sampling frequency for the above filter is 500 Hz.
# To design this filter, first determine the pole-zero location of the desired filter. Since complete rejection 250
# of signals is required at DC i.e. 0H z and 250H z, we need to place the zeros at an angle 0° and 2π*250/500=π
# on the unit circle. To have a narrow passband centered at 125 Hz, we need to place poles at 2π*125/500=π/2
# Now in order to ensure that the filter coefficients are real, we need to have a complex conjugate pole pair +/- π/2
#modulus, r = 1-(bw/fs)*pi, then the filter eqn:
# H(z) =(z-1)(z+1)/(2-re^jπ/2)(z - re^-jπ/2) = 1-z^-2/1+0.877969z^-2

fs=500
fl=0
fu=250
fp=125
bw=10
r = 1-(bw/fs)*np.pi
omeg1=2*np.pi*(fl/fs)
omeg2=2*np.pi*(fu/fs)
omegp=2*np.pi*(fp/fs)
#no calc process, if all values are given, this is better
z1=np.exp(1j*omeg1)
z2=np.exp(1j*omeg2)
p1=r*np.exp(1j*omegp)
p2=r*np.exp(-1j*omegp)
zeros=np.array([z1,z2])
poles=np.array([p1,p2])
num=np.poly(zeros).real
den=np.poly(poles).real
#calc process
# num=np.array([1,0,-1])
# den=np.array([1,0,0.877969])
w,h=signal.freqz(num,den,worN=256,fs=fs)
sys=ct.tf(num,den,dt=1/fs)
plt.subplot(3,1,1)
plt.plot(w,np.abs(h)/np.max(np.abs(h)))
plt.title('normalized frequency response')
plt.xlabel('frequency(hz)')
plt.ylabel('magnitude(normalized)')
plt.grid()
plt.subplot(3,1,2)
plt.plot(w,np.angle(h,deg=True),color='green')
plt.xlabel('frequency')
plt.ylabel('phase(deg)')
plt.title('phase response')
plt.grid()
plt.subplot(3,1,3)
ct.pzmap(sys)
plt.tight_layout()
plt.show()

################################
#FILTER DESIGN USING BLT
################################
fc=30
fs=150
wd=2*np.pi*fc
#wa=2/Ts*tan(wd*Ts/2) ========>PREWARPING THE DIGITAL CUTOFF
wa=2*fs*np.tan(wd/(2*fs))
#lpf tf=1/1+s
num=[1]
den=[1,1]
num_hpa,den_hpa=signal.lp2hp(num,den,wa)
num_hp,den_hp=signal.bilinear(num_hpa,den_hpa,fs)
w,h=signal.freqz(num_hp,den_hp,worN=512,fs=fs)
plt.subplot(2,1,1)
plt.plot(w,h)
plt.xlabel('freq')
plt.ylabel('mag')
plt.grid()
plt.subplot(2,1,2)
plt.plot(w,np.angle(h,deg=True))
plt.xlabel('freq')
plt.ylabel('phase')
plt.grid()
plt.tight_layout()
plt.show()

########################
#blt butterworth
########################

fs=20000
fpass=2000
fstop=5000
Ap=3
As=20
t=1.0/fs
wpa=(2/t)*np.tan(np.pi*fpass/fs)
wst=(2/t)*np.tan(np.pi*fstop/fs)
n,wc=signal.buttord(wpa,wst,Ap,As,analog=True)
b,a=signal.butter(n,fpass,btype='lowpass',analog=False,output='ba',fs=fs)
w,h=signal.freqz(b,a,fs=fs)
plt.plot(w/1000,20*np.log10(np.abs(h)))
plt.xlabel('frequency')
plt.ylabel('mag in db')
plt.grid()
plt.xlim(0,fs/2000)
plt.ylim(-40,5)
plt.plot(fpass/1000, -Ap, 'go',label=f'Passband Edge ({fpass/1000:.1f} kHz, -{Ap} dB)')
plt.axvline(fpass / 1000, color='g', linestyle='--')
plt.axhline(-Ap, color='g', linestyle='--')
plt.plot(fstop / 1000, -As, 'ro',label=f'Stopband Edge ({fstop/1000:.1f} kHz, -{As} dB)')
plt.axvline(fstop / 1000, color='r', linestyle='--')
plt.axhline(-As, color='r', linestyle='--')
plt.legend()
plt.show()

##############
#blt chebyshev
##############

fs=8000
Ap=3
As=40
fp1=1000
fp2=3000
fs1=1500
fs2=2500
Nyquist = fs / 2.0
wp = [fp1 / Nyquist, fp2 / Nyquist]
ws = [fs1 / Nyquist, fs2 / Nyquist]
n,wn=signal.cheb2ord(wp=wp,ws=ws,gpass=Ap,gstop=As,analog=False)
b,a=signal.cheby2(N=n,Wn=wn,rs=As,btype='bandstop',analog=False,output='ba')
w,h=signal.freqz(b,a,fs=fs)
plt.plot(w/1000,20*np.log10(np.abs(h)))
plt.xlabel('freq')
plt.ylabel('mag in db')
plt.grid()
plt.xlim(0,Nyquist/1000)
plt.ylim(-60,5)
plt.axhline(-Ap,linestyle='--',label=f'Max Passband Loss (-{Ap}dB)')
plt.axhline(-As,linestyle='--',label=f'Equiripple Stopband Floor (-{As} dB)')
plt.fill_between([fs1 / 1000, fs2 / 1000], -60, -As, color='pink',label='Equiripple Stopband Region')
plt.axvline(fp1/1000,linestyle='--')
plt.axvline(fp2/1000,linestyle='--',label='Passband Edges')
plt.legend()
plt.show()