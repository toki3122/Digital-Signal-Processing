import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import signal
# Plot the magnitude and phase responses for a 50th order Highpass filter assuming a sampling frequency
# of 5KHz and a cut-off frequency of 1KHz for each of the windows mentioned in the lab manual.
order=50
len=51
fs=5000
fc=1000
h1=signal.firwin(numtaps=len,cutoff=fc,window='rectangular',fs=fs,pass_zero='highpass')
h2=signal.firwin(numtaps=len,cutoff=fc,window='hann',fs=fs,pass_zero='highpass')
h3=signal.firwin(numtaps=len,cutoff=fc,window='hamming',fs=fs,pass_zero='highpass')
w,H1=signal.freqz(h1,1,worN=512,fs=fs)
w,H2=signal.freqz(h2,1,worN=512,fs=fs)
w,H3=signal.freqz(h3,1,worN=512,fs=fs)
plt.plot(w,20*np.log10(np.abs(H1)),label='rectangular')
plt.plot(w,20*np.log10(np.abs(H2)),label='hann')
plt.plot(w,20*np.log10(np.abs(H3)),label='hamming')
plt.axvline(fc,linestyle='--')
plt.grid()
plt.legend()
plt.show()

# Design a Band-Stop FIR Filter that is specifically intended to reject (or notch out) the frequencies between 45 Hz and 55 Hz (often used to eliminate 50 Hz or 60 Hz power line noise).
# - Filter Type: Band-Stop
# - Stopband Frequencies: 45 Hz to 55 Hz
# - Sampling Frequency: 300 Hz
# - Filter Order: 120
# - Window Type: Kaiser window (requires specifying the beta parameter, B. Use B = 8.0 for high
# stopband rejection).

Fs = 300
order = 120
numtaps = order + 1
stopband = [45, 55]
beta = 8.0
b = signal.firwin(
 numtaps,
 stopband,
 window=('kaiser', beta),
 pass_zero='bandstop',
 fs=Fs
)
f, h = signal.freqz(b, worN=4096, fs=Fs)
magnitude = 20*np.log10(np.maximum(abs(h), 1e-10))
phase = np.unwrap(np.angle(h))
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(f, magnitude)
plt.axvline(45, linestyle='--')
plt.axvline(55, linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('120th Order Kaiser Band-Stop FIR Filter')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(f, phase)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (rad)')
plt.title('Phase Response')
plt.grid()
plt.tight_layout()
plt.show() 