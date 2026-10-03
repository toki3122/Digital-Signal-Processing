"""
===============================================================================
ECG Compression – Beginner-Friendly Version
===============================================================================
This script shows how to compress ECG signals by reducing the number of bits
used to store each sample (quantization) and then reconstruct the signal with
a digital low-pass filter.

Everything is written so that a student who has only studied the EEE 3218
Digital Signal Processing Laboratory manual can understand the code.

Related topics from the lab manual
----------------------------------
Lab 2  – Sampling and Quantization of Signals
         (mid-tread quantizer, quantization levels, reconstruction)
Lab 3  – Generation and Operation on Discrete-Time Sequences
         (amplitude scaling, time shifting, signal addition)
Lab 5/6 – Digital Filter Design (FIR & IIR)
         (Butterworth IIR, window-method FIR, filtering with lfilter / filtfilt)
Lab 7/8 – Application of Signal Processing
         (denoising, feature extraction)

The algorithms (quantizers + reconstruction filters) are exactly the same as
in the original research code. Only the style, variable names and comments
have been simplified.
"""

# ---------------------------------------------------------------------------
# 1. Import the libraries we need
# ---------------------------------------------------------------------------
import math
import os
import io
import zipfile
from functools import partial

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 2. Global constants (easy to change)
# ---------------------------------------------------------------------------
FS = 360                # Sampling frequency of MIT-BIH ECG records (Hz)
                        # (Lab 2: sampling rate)

RES = 11                # Original resolution of the ECG samples (bits)
                        # Each sample is an integer from 0 to 2047

DURATION_S = 300        # How many seconds of each record we use

# Bit depths we will test (from 1-bit up to 10-bit)
BITS_SWEEP = list(range(1, 11))

# A few bit depths we will highlight in the reconstruction pictures
HIGHLIGHT_BITS = [2, 4, 6, 10]

# Different sampling rates we will try after quantization (for the paper figures)
RESAMPLE_RATES = [360, 180, 120]

BITS_MAIN = 4           # The bit depth used for most comparison tables

# Low-pass filter parameters used for reconstruction
# (Lab 5/6: filter design)
FC = 27.0               # Cut-off frequency in Hz
ORDER = 3               # Order of the Butterworth IIR filter
FIR_TAPS = 101          # Number of coefficients of the FIR filter

# Folder where all results and plots will be saved
OUT_DIR = "ecg_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Design the two reconstruction filters once (Lab 5/6)
# Butterworth IIR (zero-phase version will use filtfilt)
BUTT_B, BUTT_A = signal.butter(ORDER, FC, fs=FS)

# Window-method FIR (Lab 5: FIR filter design using window method)
FIR_H = signal.firwin(FIR_TAPS, FC, window="hamming", fs=FS)

# List of all 48 MIT-BIH arrhythmia records
RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

# Records that contain pacemaker spikes (we skip them for QRS detection)
PACEMAKER_RECORDS = {"102", "104", "107", "217"}

# Symbols that mean a real heartbeat in the MIT-BIH annotation files
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


# ---------------------------------------------------------------------------
# 3. Loading ECG data
# ---------------------------------------------------------------------------
def load_mitbih(name, seconds):
    """
    Load one MIT-BIH record from PhysioNet.
    Returns the raw integer samples and the locations of the R-peaks.
    """
    import wfdb
    n = int(seconds * FS)
    rec = wfdb.rdrecord(name, pn_dir="mitdb", channels=[0],
                        physical=False, sampto=n)
    ann = wfdb.rdann(name, "atr", pn_dir="mitdb", sampto=n)

    # Convert to ordinary Python integers (0 … 2047)
    x = rec.d_signal[:, 0].astype(np.int64)

    # Keep only the annotation symbols that are real beats
    beats = np.array([s for s, sym in zip(ann.sample, ann.symbol)
                      if sym in BEAT_SYMBOLS])
    return x, beats


def synth_ecg(seconds, seed):
    """
    Create a simple synthetic ECG when real data cannot be downloaded.
    Useful for testing the rest of the code offline.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS

    # Random heart rate between 60 and 85 bpm
    hr = rng.uniform(60, 85)

    # Generate R-peak times
    r_times = []
    tt = 0.6
    while tt < seconds - 0.6:
        r_times.append(tt)
        tt += 60.0 / hr + rng.normal(0, 0.03)

    # Simple Gaussian-shaped P, Q, R, S, T waves
    waves = [(-0.20, 0.12, 0.025),   # P
             (-0.04, -0.15, 0.010),  # Q
             (0.00, 1.00, 0.012),    # R
             (0.04, -0.25, 0.012),   # S
             (0.28, 0.30, 0.050)]    # T

    mv = np.zeros(n)
    for r in r_times:
        for off, amp, width in waves:
            mv += amp * np.exp(-0.5 * ((t - r - off) / width) ** 2)

    # Add a little baseline wander and noise
    mv += 0.10 * np.sin(2 * np.pi * 0.25 * t)
    mv += rng.normal(0, 0.012, n)

    # Convert to the same 11-bit integer range used by MIT-BIH
    x = np.clip(np.rint(1024 + 250 * mv), 0, 2047).astype(np.int64)
    beats = np.rint(np.array(r_times) * FS).astype(int)
    return x, beats


def get_data(seconds=DURATION_S):
    """
    Try to load real MIT-BIH data.
    If that fails, fall back to synthetic ECG.
    """
    try:
        data = [(r, *load_mitbih(r, seconds)) for r in RECORDS]
        print(f"DATA: MIT-BIH via PhysioNet, {len(data)} records × {seconds}s")
        return data, "MIT-BIH (PhysioNet)"
    except Exception as e:
        print(f"[data] PhysioNet unavailable ({type(e).__name__})")
        print("       Falling back to SYNTHETIC ECG")

    data = [(f"syn{i}", *synth_ecg(seconds, seed=i))
            for i in range(len(RECORDS))]
    print(f"DATA: SYNTHETIC, {len(data)} records × {seconds}s @ {FS} Hz")
    return data, "SYNTHETIC"


# ---------------------------------------------------------------------------
# 4. Quantization step size
# ---------------------------------------------------------------------------
def step_size(b):
    """
    Calculate the quantization step size for b-bit quantization.

    Original signal has RES = 11 bits → values 0 … 2047.
    When we keep only b bits the step becomes 2^(11-b).

    Example (Lab 2 – Quantization Levels):
        b = 11 → step = 1
        b = 4  → step = 128
        b = 1  → step = 1024
    """
    return 1 << (RES - b)          # same as 2**(RES - b)


# ---------------------------------------------------------------------------
# 5. The different quantizers
# ---------------------------------------------------------------------------
def quantize_plain(x, b):
    """
    Classic mid-tread quantizer (Lab 2 – Mid-Tread Quantizer).

    Formula used in the lab:
        q = floor( x / S + 0.5 )
    where S is the step size.
    """
    S = step_size(b)
    max_code = (1 << b) - 1          # highest possible code (2^b - 1)
    codes = np.floor(x / S + 0.5)
    codes = np.clip(codes, 0, max_code)
    return codes.astype(np.int64)


def quantize_paper(x, b):
    """
    Algorithm 1 from the paper (floor quantizer + error accumulator).

    Idea:
    - First quantize by simple floor division.
    - Keep a running sum of the quantization error.
    - Whenever the accumulated error becomes larger than one step,
      “steal” one extra step for the current sample.
    """
    S = step_size(b)
    limit = 1 << RES                 # 2048
    codes = np.empty(len(x), dtype=np.int64)
    accumulated_error = 0

    for i, sample in enumerate(x):
        # Floor quantization
        yq = (sample // S) * S
        accumulated_error += sample - yq

        # If we have collected enough error, give this sample one extra step
        if accumulated_error > S and yq + S < limit:
            yq += S
            accumulated_error -= S

        codes[i] = yq // S

    return codes


def quantize_error_feedback(x, b, order=2, use_dither=False, seed=0):
    """
    Error-feedback (noise-shaping) quantizer.

    order = 1 → first-order error feedback
    order = 2 → second-order noise shaping  (1 - z^-1)^2

    Optional TPDF dither can be added before rounding
    (Lab 2 – Quantization noise).
    """
    S = step_size(b)
    max_code = (1 << b) - 1
    codes = np.empty(len(x), dtype=np.int64)

    # Optional dither (two uniform random numbers → triangular pdf)
    if use_dither:
        rng = np.random.default_rng(seed)
        dither = (rng.random(len(x)) + rng.random(len(x)) - 1.0) * S
    else:
        dither = np.zeros(len(x))

    e1 = 0.0   # previous quantization error
    e2 = 0.0   # error before that (needed for order-2)

    for i, sample in enumerate(x):
        # Form the input to the quantizer according to the order
        if order == 0:
            u = sample
        elif order == 1:
            u = sample - e1
        else:                       # order == 2
            u = sample - 2*e1 + e2

        # Mid-tread quantize (with optional dither)
        code = int(math.floor((u + dither[i]) / S + 0.5))
        code = max(0, min(max_code, code))

        # Quantization error (limited to ±S)
        error = max(-S, min(S, code * S - u))

        # Shift the error history
        e2 = e1
        e1 = error

        codes[i] = code

    return codes


def dpcm_encode(x, b, delta):
    """
    Simple Differential Pulse-Code Modulation (DPCM).

    Instead of quantizing the sample itself we quantize the
    difference between the current sample and a prediction
    (here the prediction is just the previous reconstructed sample).

    This is a classic predictive coding technique.
    """
    half = 1 << (b - 1)              # codes are stored as unsigned
    prev = float(x[0])
    reconstructed = [prev]
    codes = [half]                   # first sample coded as “zero difference”

    for sample in x[1:]:
        # Prediction error
        diff = sample - prev
        code = int(math.floor(diff / delta + 0.5))
        # Clip to the representable range
        code = max(-half, min(half - 1, code))

        # Reconstruct
        prev = prev + code * delta
        prev = max(0.0, min(2047.0, prev))

        reconstructed.append(prev)
        codes.append(code + half)    # store as unsigned

    return np.array(codes, dtype=np.int64), np.array(reconstructed)


def quantize_dpcm(x, b):
    """
    Try several step sizes (delta) and keep the one that gives
    the smallest PRD.  This is a simple automatic tuning.
    """
    best = None
    for delta in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        codes, recon = dpcm_encode(x, b, delta)
        p = percent_rms_difference(x, recon, remove_dc=True)
        if best is None or p < best[0]:
            best = (p, codes, recon, delta)
    return best[1], best[2], best[3]


# Dictionary that maps a friendly name to the corresponding quantizer
QUANTIZERS = {
    "plain":          quantize_plain,
    "paper Alg.1":    quantize_paper,
    "EF-1 round":     partial(quantize_error_feedback, order=1),
    "EF-2 round":     partial(quantize_error_feedback, order=2),
    "dither+plain":   partial(quantize_error_feedback, order=0, use_dither=True),
}

METHODS = list(QUANTIZERS) + ["DPCM"]


# ---------------------------------------------------------------------------
# 6. Reconstruction (digital low-pass filtering)
# ---------------------------------------------------------------------------
def iir_causal(v):
    """
    Causal IIR filtering with scipy.signal.lfilter.
    We initialise the filter state so that there is no big transient
    at the beginning (steady-state start).
    """
    zi = signal.lfilter_zi(BUTT_B, BUTT_A) * v[0]
    y, _ = signal.lfilter(BUTT_B, BUTT_A, v, zi=zi)
    return y


def fir_delay_compensated(v):
    """
    Causal FIR filtering + exact group-delay compensation.

    A linear-phase FIR of length N has a constant delay of (N-1)/2 samples.
    We remove that delay so the reconstructed waveform lines up with
    the original.
    """
    delay = (FIR_TAPS - 1) // 2
    # Pad the signal so the filter can “see” past the ends
    padded = np.concatenate([
        np.full(FIR_TAPS, v[0]),
        v,
        np.full(delay, v[-1])
    ])
    y = signal.lfilter(FIR_H, 1.0, padded)
    # Remove the exact delay
    return y[FIR_TAPS + delay : FIR_TAPS + delay + len(v)]


def align_by_lag(original, filtered, max_lag=40):
    """
    Find the integer lag that minimises the PRD between the original
    and the filtered signal (simple delay compensation).
    """
    best_prd = np.inf
    best_y = filtered
    best_lag = 0

    for lag in range(max_lag + 1):
        if lag == 0:
            y = filtered
        else:
            y = np.concatenate([filtered[lag:], np.full(lag, filtered[-1])])

        p = percent_rms_difference(original[:-max_lag], y[:-max_lag],
                                   remove_dc=True)
        if p < best_prd:
            best_prd = p
            best_y = y
            best_lag = lag

    return best_y, best_lag


def reconstruct_variants(original, raw_quantized):
    """
    Produce four different reconstructions of the same quantized signal:

    1. filtfilt          – zero-phase Butterworth (non-causal)
    2. iir_causal_raw    – ordinary causal IIR (has delay)
    3. iir_causal_aln    – causal IIR + integer lag compensation
    4. fir_delay_comp    – causal FIR with exact delay removed
    """
    raw = raw_quantized.astype(float)

    # 1. Zero-phase IIR (Lab 6 – filtfilt)
    y_filtfilt = signal.filtfilt(BUTT_B, BUTT_A, raw)

    # 2. Causal IIR without any delay compensation
    y_causal = iir_causal(raw)

    # 3. Causal IIR + best integer lag
    y_aligned, lag = align_by_lag(original, y_causal)

    # 4. Linear-phase FIR with known delay removed
    y_fir = fir_delay_compensated(raw)

    variants = {
        "filtfilt":        y_filtfilt,
        "iir_causal_raw":  y_causal,
        "iir_causal_aln":  y_aligned,
        "fir_delay_comp":  y_fir,
    }
    return variants, lag


# ---------------------------------------------------------------------------
# 7. Quality metrics
# ---------------------------------------------------------------------------
def percent_rms_difference(x, y, remove_dc=False):
    """
    Percent Root-mean-square Difference (PRD) – the most common
    quality measure for ECG compression.

    PRD = 100 * sqrt( sum( (x-y)^2 ) / sum( x^2 ) )

    When remove_dc=True we first subtract the mean of x from both
    signals (this removes the effect of a constant baseline offset).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if remove_dc:
        mean_x = x.mean()
        x = x - mean_x
        y = y - mean_x

    numerator = np.sum((x - y) ** 2)
    denominator = np.sum(x ** 2)
    return 100.0 * math.sqrt(numerator / denominator)


def entropy_bits(codes):
    """
    Estimate the average number of bits needed to store the codes
    (Shannon entropy).  Lower entropy means the codes are more
    compressible by a later entropy coder (Huffman, arithmetic, …).
    """
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts / counts.sum()
    return float( -(probabilities * np.log2(probabilities)).sum() )


def pack_into_64bit_words(codes, b):
    """
    Pack many b-bit codes into 64-bit integers.
    This is how the compressed bitstream would be stored on disk.
    """
    codes_per_word = 64 // b
    # Pad so the length is a multiple of codes_per_word
    pad = (-len(codes)) % codes_per_word
    padded = np.concatenate([codes, np.zeros(pad, dtype=np.int64)])
    padded = padded.astype(np.uint64).reshape(-1, codes_per_word)

    shifts = np.arange(codes_per_word, dtype=np.uint64) * np.uint64(b)
    words = np.bitwise_or.reduce(padded << shifts, axis=1)
    return words


def unpack_from_64bit_words(words, b, n):
    """
    Inverse of pack_into_64bit_words – recovers the original codes.
    """
    codes_per_word = 64 // b
    shifts = np.arange(codes_per_word, dtype=np.uint64) * np.uint64(b)
    mask = np.uint64((1 << b) - 1)
    codes = ((words[:, None] >> shifts) & mask).reshape(-1)
    return codes[:n].astype(np.int64)


def zip_compressed_size(words):
    """
    Further compress the packed words with ordinary ZIP (deflate).
    Gives an idea of the final file size after entropy coding.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        zf.writestr("r.bin", words.tobytes())
    return buf.tell()


# ---------------------------------------------------------------------------
# 8. Simple QRS detector (for diagnostic quality check)
# ---------------------------------------------------------------------------
def simple_qrs_detector(sig):
    """
    A very basic R-peak detector (band-pass → derivative → square →
    moving average → peak picking).

    This is only used to measure how well the compressed signal
    still allows correct heartbeat detection.
    """
    # Band-pass 5–15 Hz
    b, a = signal.butter(2, [5, 15], btype="band", fs=FS)
    filtered = signal.filtfilt(b, a, sig - np.mean(sig))

    # Squared derivative
    squared = np.gradient(filtered) ** 2

    # Moving average (150 ms window)
    win = int(0.15 * FS)
    ma = np.convolve(squared, np.ones(win) / win, mode="same")

    # Peak picking
    peaks, _ = signal.find_peaks(
        ma,
        height=0.25 * np.percentile(ma, 99.5),
        distance=int(0.25 * FS)
    )

    # Refine to the real R-peak inside a small window
    refined = []
    radius = int(0.1 * FS)
    for p in peaks:
        start = max(0, p - radius)
        end = p + radius
        refined.append(start + np.argmax(np.abs(filtered[start:end])))
    return np.array(refined)


def detect_qrs(sig):
    """
    Try the professional detector from the wfdb package first.
    If it is not available, fall back to the simple detector above.
    """
    try:
        import wfdb.processing as wp
        peaks = np.asarray(wp.xqrs_detect(sig=np.asarray(sig, float),
                                          fs=FS, verbose=False))
        if len(peaks) > 3:
            return peaks
    except Exception:
        pass
    return simple_qrs_detector(np.asarray(sig, float))


def match_peaks(reference, detected, tolerance=None):
    """
    Count true positives, false negatives and false positives
    so we can compute Sensitivity and Positive Predictivity.
    """
    if tolerance is None:
        tolerance = int(0.1 * FS)          # 100 ms

    reference = np.sort(reference)
    detected = np.sort(detected)
    used = np.zeros(len(detected), dtype=bool)
    true_positives = 0

    for r in reference:
        # Look for the nearest unused detected peak inside the tolerance
        idx = np.searchsorted(detected, r)
        candidates = []
        for k in (idx - 1, idx):
            if 0 <= k < len(detected) and not used[k]:
                if abs(detected[k] - r) <= tolerance:
                    candidates.append(k)
        if candidates:
            best = min(candidates, key=lambda k: abs(detected[k] - r))
            used[best] = True
            true_positives += 1

    false_negatives = len(reference) - true_positives
    false_positives = len(detected) - true_positives

    sensitivity = 100.0 * true_positives / max(true_positives + false_negatives, 1)
    positive_predictivity = 100.0 * true_positives / max(true_positives + false_positives, 1)
    return sensitivity, positive_predictivity


# ---------------------------------------------------------------------------
# 9. Encode + decode one record
# ---------------------------------------------------------------------------
def encode_and_decode(x, b, method):
    """
    High-level helper:
    - quantize the signal with the chosen method
    - reconstruct it with the different filters
    - return everything needed for later analysis
    """
    if method == "DPCM":
        codes, reconstructed, delta = quantize_dpcm(x, b)
        variants = {"direct": reconstructed}
        note = f"delta={delta}"
        raw = None
        return codes, variants, note, raw

    # All other methods produce integer codes that must be scaled back
    codes = QUANTIZERS[method](x, b)
    raw = codes * step_size(b)                 # back to the original amplitude scale
    variants, lag = reconstruct_variants(x, raw)
    note = f"lag={lag}"
    return codes, variants, note, raw


# ---------------------------------------------------------------------------
# 10. Main experimental loops
# ---------------------------------------------------------------------------
def run_bit_depth_sweep(data):
    """
    For every record, every bit depth and every quantizer,
    compute PRD, bit-rate, entropy, etc.
    """
    rows = []
    keep_for_plots = {}

    for record_name, x, beats in data:
        n = len(x)
        for b in BITS_SWEEP:
            for method in METHODS:
                codes, variants, note, raw = encode_and_decode(x, b, method)

                # Pack the codes into 64-bit words (simulates the bitstream)
                words = pack_into_64bit_words(codes, b)
                # Sanity check – unpacking must give the original codes
                assert np.array_equal(unpack_from_64bit_words(words, b, n), codes)

                packed_bytes = words.nbytes

                for variant_name, y in variants.items():
                    prd_raw = percent_rms_difference(x, y)
                    prd_dc  = percent_rms_difference(x, y, remove_dc=True)

                    rows.append({
                        "record":   record_name,
                        "bits":     b,
                        "method":   method,
                        "variant":  variant_name,
                        "note":     note,
                        "PRD_raw":  prd_raw,
                        "PRD_dc":   prd_dc,
                        "BCR_vs16": 2 * n / packed_bytes,
                        "BCR_vs11": (RES * n / 8) / packed_bytes,
                        "H_bits":   entropy_bits(codes),
                        "save_vs16": 100 * (1 - packed_bytes / (2 * n)),
                        "save_zip":  100 * (1 - zip_compressed_size(words) / (2 * n)),
                        "QS_raw":   (2 * n / packed_bytes) / max(prd_raw, 1e-6),
                        "QS_dc":    (2 * n / packed_bytes) / max(prd_dc, 1e-6),
                    })

                # Keep a few signals for the time-domain plot
                if record_name == data[0][0] and b == BITS_MAIN:
                    keep_for_plots[method] = {
                        "codes": codes,
                        "variants": variants,
                        "raw": raw
                    }

        print(f"  finished record {record_name}")

    return pd.DataFrame(rows), keep_for_plots


def run_qrs_evaluation(data):
    """
    Check how well a simple QRS detector still works on the
    reconstructed signals (diagnostic quality).
    """
    rows = []
    for record_name, x, beats in data:
        if record_name in PACEMAKER_RECORDS:
            continue

        # Original 11-bit signal
        d_orig = detect_qrs(x)
        se, pp = match_peaks(beats, d_orig)
        rows.append({
            "record": record_name,
            "method": "ORIGINAL 11-bit",
            "variant": "-",
            "SE": se,
            "PP": pp,
            "HV_samples": 0.0
        })

        for method in METHODS:
            _, variants, _, _ = encode_and_decode(x, BITS_MAIN, method)
            variants_to_test = ["direct"] if method == "DPCM" else ["filtfilt", "fir_delay_comp"]
            for v in variants_to_test:
                d = detect_qrs(variants[v])
                se, pp = match_peaks(beats, d)
                rows.append({
                    "record": record_name,
                    "method": method,
                    "variant": v,
                    "SE": se,
                    "PP": pp,
                    "HV_samples": np.nan   # HRV error omitted for simplicity
                })
    return pd.DataFrame(rows)


def reconstruct_at_rate(codes, b, rate, n_original):
    """
    Optional extra experiment from the paper:
    after quantization we can also lower the sampling rate,
    then interpolate back to the original rate before filtering.
    """
    S = step_size(b)
    if rate == FS:
        raw = codes.astype(float) * S
        return signal.filtfilt(BUTT_B, BUTT_A, raw), len(codes)

    duration = (len(codes) - 1) / FS
    # Linear interpolation (Lab 2 – Signal Reconstruction)
    t_src = np.arange(len(codes)) / FS
    n_dst = int(round(duration * rate)) + 1
    t_dst = np.arange(n_dst) / rate
    down = np.interp(t_dst, t_src, codes.astype(float))

    # Upsample back to original length
    t_orig = np.arange(n_original) / FS
    up = np.interp(t_orig, t_dst, down)
    raw = up * S
    return signal.filtfilt(BUTT_B, BUTT_A, raw), len(down)


def run_resample_sweep(data):
    """
    Reproduce the paper’s figures that show PRD and bit-rate
    for different combinations of bit depth and sampling rate.
    """
    rows = []
    for record_name, x, beats in data:
        for b in BITS_SWEEP:
            codes = quantize_paper(x, b)          # only the paper algorithm
            for rate in RESAMPLE_RATES:
                y, n_stored = reconstruct_at_rate(codes, b, rate, len(x))
                prd = percent_rms_difference(x, y)
                bcr = 16 * FS / (b * rate)        # bits of original / bits stored
                rows.append({
                    "record": record_name,
                    "bits": b,
                    "rate": rate,
                    "PRD_raw": prd,
                    "BCR_vs16": bcr,
                    "n_stored": n_stored
                })
        print(f"  resample-swept record {record_name}")
    return pd.DataFrame(rows)


def run_qrs_vs_bits(data):
    """
    How does QRS detection performance change when we use fewer bits
    or a lower sampling rate?
    """
    rows = []
    for record_name, x, beats in data:
        if record_name in PACEMAKER_RECORDS:
            continue
        for b in BITS_SWEEP:
            codes = quantize_paper(x, b)
            for rate in RESAMPLE_RATES:
                y, _ = reconstruct_at_rate(codes, b, rate, len(x))
                d = simple_qrs_detector(y)
                se, pp = match_peaks(beats, d)
                rows.append({
                    "record": record_name,
                    "bits": b,
                    "rate": rate,
                    "SE": se,
                    "PP": pp
                })
        print(f"  qrs-vs-bits record {record_name}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 11. Plotting functions (identical results to the original code)
# ---------------------------------------------------------------------------
def make_main_plots(df, keep, data):
    """
    Three standard plots:
    1. PRD versus number of bits
    2. Power spectrum of the quantization error
    3. Time-domain overlay of original and reconstructed signals
    """
    name, x, _ = data[0]
    sel = df[df.variant.isin(["filtfilt", "direct"])]
    g = sel.groupby(["method", "bits"])[["PRD_raw", "PRD_dc"]].mean().reset_index()

    # ---- Plot 1: PRD vs bits ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in METHODS:
        s = g[g.method == m]
        ax[0].semilogy(s.bits, s.PRD_raw, "o-", label=m)
        ax[1].semilogy(s.bits, s.PRD_dc, "o-", label=m)
    ax[0].set_title("PRD (raw, as in paper)")
    ax[1].set_title("PRD (DC removed)")
    for a in ax:
        a.set_xlabel("bits")
        a.set_ylabel("PRD %")
        a.grid(True, which="both", ls=":")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_prd_vs_bits.png", dpi=130)

    # ---- Plot 2: Error spectrum ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in ["plain", "paper Alg.1", "EF-1 round", "EF-2 round"]:
        e = keep[m]["raw"] - x
        f, P = signal.welch(e - e.mean(), fs=FS, nperseg=1024)
        ax.semilogy(f, P, label=m)
    ax.axvline(FC, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Hz")
    ax.set_ylabel("PSD of quantization error")
    ax.set_title(f"Error spectrum before low-pass, {BITS_MAIN} bits (dashed = {FC:.0f} Hz)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_error_spectrum.png", dpi=130)

    # ---- Plot 3: Time-domain overlay ----------------------------------------
    i0 = 3 * FS
    sl = slice(i0, i0 + 2 * FS)
    t = np.arange(len(x))[sl] / FS
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x[sl], "k", lw=1.5, label="original 11-bit")
    ax.plot(t, keep["plain"]["variants"]["filtfilt"][sl], lw=1, label="plain + LPF")
    ax.plot(t, keep["paper Alg.1"]["variants"]["filtfilt"][sl], lw=1, label="paper + LPF")
    ax.plot(t, keep["DPCM"]["variants"]["direct"][sl], lw=1, label="DPCM")
    ax.set_title(f"{BITS_MAIN}-bit reconstructions ({name})")
    ax.set_xlabel("s")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_overlay.png", dpi=130)
    plt.show()


def fig_reconstruction_grid(data):
    """
    2×2 grid showing original vs reconstructed ECG
    at 2, 4, 6 and 10 bits (paper Algorithm 1).
    """
    name, x, _ = data[0]
    i0 = 3 * FS
    sl = slice(i0, i0 + 250)
    t = np.arange(len(x))[sl]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, b in zip(axes.flat, HIGHLIGHT_BITS):
        codes = quantize_paper(x, b)
        y, _ = reconstruct_at_rate(codes, b, FS, len(x))
        ax.plot(t, x[sl], "b", lw=1, label="Original")
        ax.plot(t, y[sl], "r", lw=1, label="Reconstructed")
        ax.set_title(f"{b}-bit")
        ax.set_xlabel("Sample points")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7)
    fig.suptitle(f"Original vs reconstructed at different bit depths "
                 f"(record {name}, paper Alg.1)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_reconstruction_grid.png", dpi=130)


def fig_prd_bcr_vs_bits(resample_df):
    """
    Paper’s Figure 5: PRD and Bit Compression Ratio versus number of bits
    for three different sampling rates.
    """
    g = resample_df.groupby(["rate", "bits"])[["PRD_raw", "BCR_vs16"]].mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for rate in RESAMPLE_RATES:
        s = g[g.rate == rate].sort_values("bits")
        ax[0].plot(s.bits, s.PRD_raw, "o-", label=f"{rate} Hz")
        ax[1].plot(s.bits, s.BCR_vs16, "o-", label=f"{rate} Hz")
    ax[0].set_title("(a) PRD vs number of bits")
    ax[0].set_ylabel("PRD (%)")
    ax[1].set_title("(b) BCR vs number of bits")
    ax[1].set_ylabel("BCR")
    for a in ax:
        a.set_xlabel("Number of bits")
        a.grid(True, ls=":")
        a.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_prd_bcr_vs_bits.png", dpi=130)


def fig_qrs_vs_bits(qrsbits_df):
    """
    Paper’s Figure 6: Sensitivity and Positive Predictivity of the
    QRS detector versus number of bits.
    """
    g = qrsbits_df.groupby(["rate", "bits"])[["SE", "PP"]].mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for rate in RESAMPLE_RATES:
        s = g[g.rate == rate].sort_values("bits")
        ax[0].plot(s.bits, s.SE, "o-", label=f"{rate} Hz")
        ax[1].plot(s.bits, s.PP, "o-", label=f"{rate} Hz")
    ax[0].set_title("(a) Sensitivity vs number of bits")
    ax[0].set_ylabel("Sensitivity (%)")
    ax[1].set_title("(b) Positive precision vs number of bits")
    ax[1].set_ylabel("+P (%)")
    for a in ax:
        a.set_xlabel("Number of bits")
        a.grid(True, ls=":")
        a.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6_qrs_vs_bits.png", dpi=130)


# ---------------------------------------------------------------------------
# 12. Main program
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 200,
                  "display.max_columns", 30,
                  "display.float_format", "{:.3f}".format)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data, src = get_data(DURATION_S)

    # ------------------------------------------------------------------
    # Full bit-depth sweep (all quantizers)
    # ------------------------------------------------------------------
    df, keep = run_bit_depth_sweep(data)
    df.to_csv(f"{OUT_DIR}/sweep_all.csv", index=False)

    # Summary table for the main bit depth
    main = (df[df.bits == BITS_MAIN]
            .groupby(["method", "variant"])
            .agg(PRD_raw=("PRD_raw", "mean"),
                 PRD_dc=("PRD_dc", "mean"),
                 BCR_vs16=("BCR_vs16", "mean"),
                 BCR_vs11=("BCR_vs11", "mean"),
                 H_bits=("H_bits", "mean"),
                 save_vs16=("save_vs16", "mean"),
                 save_zip=("save_zip", "mean"),
                 QS_raw=("QS_raw", "mean"),
                 QS_dc=("QS_dc", "mean"))
            .reset_index())

    print(f"\n=== {BITS_MAIN}-bit results (mean of per-record PRD) "
          f"over {len(data)} records ({src}) ===")
    print(main.sort_values("PRD_dc").to_string(index=False))
    main.to_csv(f"{OUT_DIR}/table_main.csv", index=False)

    # ------------------------------------------------------------------
    # QRS detection quality
    # ------------------------------------------------------------------
    print(f"\n=== QRS detection at {BITS_MAIN} bits ===")
    q = (run_qrs_evaluation(data)
         .groupby(["method", "variant"])[["SE", "PP"]]
         .mean()
         .reset_index())
    print(q.to_string(index=False))
    q.to_csv(f"{OUT_DIR}/table_qrs.csv", index=False)

    # ------------------------------------------------------------------
    # Paper figures (resample experiments)
    # ------------------------------------------------------------------
    print(f"\n=== Paper Fig. 5 – PRD & BCR vs bits ===")
    resample_df = run_resample_sweep(data)
    resample_df.to_csv(f"{OUT_DIR}/table_prd_bcr_vs_bits.csv", index=False)
    print(resample_df.groupby(["rate", "bits"])[["PRD_raw", "BCR_vs16"]]
          .mean().reset_index().to_string(index=False))
    fig_prd_bcr_vs_bits(resample_df)

    print(f"\n=== Paper Fig. 6 – QRS SE/+P vs bits ===")
    qrsbits_df = run_qrs_vs_bits(data)
    qrsbits_df.to_csv(f"{OUT_DIR}/table_qrs_vs_bits.csv", index=False)
    print(qrsbits_df.groupby(["rate", "bits"])[["SE", "PP"]]
          .mean().reset_index().to_string(index=False))
    fig_qrs_vs_bits(qrsbits_df)

    # Reconstruction grid (2,4,6,10 bits)
    fig_reconstruction_grid(data)

    # The three classic plots
    make_main_plots(df, keep, data)

    print(f"\nAll tables and figures have been saved in ./{OUT_DIR}/")
