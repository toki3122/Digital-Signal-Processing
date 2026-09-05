import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
Fs = 5000
Fc = 1000
numtaps = 51
windows = [
    'boxcar',
    'bartlett',
    'hann',
    'hamming',
    'blackman',
    ('kaiser', 8.0)
]
names = [
    'Rectangular',
    'Bartlett',
    'Hanning',
    'Hamming',
    'Blackman',
    'Kaiser'
]
for window, name in zip(windows, names):
    b = signal.firwin(
        numtaps,
        Fc,
        window=window,
        pass_zero='highpass',
        fs=Fs
    )
    f, h = signal.freqz(b, worN=2048, fs=Fs)
    magnitude = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    phase = np.unwrap(np.angle(h))
    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(f, magnitude)
    plt.axvline(Fc, linestyle='--')
    plt.title(name + ' - Magnitude Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(f, phase)
    plt.axvline(Fc, linestyle='--')
    plt.title(name + ' - Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    plt.grid()
    plt.tight_layout()
    plt.show()