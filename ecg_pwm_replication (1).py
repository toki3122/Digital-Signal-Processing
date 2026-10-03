"""
ECG compression: replication of Devindi et al. 2024 (Scientific Reports) doi: https://doi.org/10.1038/s41598-024-68022-5
Methods (all quantize 11-bit samples to `b` bits, stored codes are b bits):
  plain (mid-tread quantizer) - mid tread quantizer + low-pass (Baseline for comparison)          
  paper Alg.1               floor quantizer + error accumulator (the paper method)
  EF-1 round                1st-order error feedback with rounding
  EF-2 round                2nd-order noise shaping, TF=(1-z^-1)^2
  dither+plain              TPDF dither before rounding
  DPCM                      closed-loop previous-sample predictor, b-bit residual, step tuned per record
Receiver-side reconstruction variants for the quantizer methods:
  filtfilt         3rd-order Butterworth 27 Hz, zero-phase
  iir_causal_raw   same filter with lfilter, NO delay compensation (what a naive port gives)
  iir_causal_aln   lfilter + integer-lag compensation
  fir_delay_comp   windowed-FIR (hamming, 101 taps, 27 Hz), causal, known delay removed
"""
import io
import math
import os
import zipfile
from functools import partial

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

FS = 360 # MIT-BIH sampling rate
RES = 11 
RECORDS = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112",
           "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200",
           "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215",
           "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"]
# all 48 official MIT-BIH arrhythmia records
PACEMAKER_RECORDS = {"102", "104", "107", "217"}
DURATION_S = 300          
BITS_SWEEP = list(range(1, 11))   
HIGHLIGHT_BITS = [2, 4, 6, 10]    
RESAMPLE_RATES = [360, 180, 120]
BITS_MAIN = 4
FC, ORDER = 27.0, 3 
FIR_TAPS = 101
OUT_DIR = "ecg_results"
os.makedirs(OUT_DIR, exist_ok=True)

BUTT_B, BUTT_A = signal.butter(ORDER, FC, fs=FS)
FIR_H = signal.firwin(FIR_TAPS, FC, window="hamming", fs=FS)

BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
MIRROR_BASE = "https://github.com/MIT-LCP/wfdb-python/tree/main/sample-data"
MIRROR_RECORDS = ["100"]
def fetch_mirror(name, dest_dir):
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    for ext in ("dat", "hea", "atr"):
        path = os.path.join(dest_dir, f"{name}.{ext}")
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{MIRROR_BASE}/{name}.{ext}", path)
    return os.path.join(dest_dir, name)


def load_mitbih(name, seconds):
    import wfdb
    n = int(seconds * FS)
    rec = wfdb.rdrecord(name, pn_dir="mitdb", channels=[0], physical=False, sampto=n)
    ann = wfdb.rdann(name, "atr", pn_dir="mitdb", sampto=n)
    x = rec.d_signal[:, 0].astype(np.int64)
    beats = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS])
    return x, beats


def load_mitbih_local(path, seconds):
    import wfdb
    n = int(seconds * FS)
    rec = wfdb.rdrecord(path, channels=[0], physical=False, sampto=n)
    ann = wfdb.rdann(path, "atr", sampto=n)
    x = rec.d_signal[:, 0].astype(np.int64)
    beats = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS])
    return x, beats


def synth_ecg(seconds, seed):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    hr = rng.uniform(60, 85)
    r_times, tt = [], 0.6
    while tt < seconds - 0.6:
        r_times.append(tt)
        tt += 60 / hr + rng.normal(0, 0.03)
    waves = [(-0.20, 0.12, 0.025), (-0.04, -0.15, 0.010), (0.0, 1.0, 0.012),
             (0.04, -0.25, 0.012), (0.28, 0.30, 0.050)]
    mv = np.zeros(n)
    for r in r_times:
        for off, a, w in waves:
            mv += a * np.exp(-0.5 * ((t - r - off) / w) ** 2)
    mv += 0.10 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.012, n)
    x = np.clip(np.rint(1024 + 250 * mv), 0, 2047).astype(np.int64)
    return x, np.rint(np.array(r_times) * FS).astype(int)


def get_data(seconds=DURATION_S):
    try:
        data = [(r, *load_mitbih(r, seconds)) for r in RECORDS]
        print(f"DATA: MIT-BIH via PhysioNet, {len(data)} record(s) x {seconds}s @ {FS} Hz")
        return data, "MIT-BIH (PhysioNet)"
    except Exception as e:
        print(f"[data] PhysioNet unavailable ({type(e).__name__}); trying GitHub mirror")
    try:
        data = []
        for r in MIRROR_RECORDS:
            path = fetch_mirror(r, os.path.join(OUT_DIR, "mirror-data"))
            data.append((r, *load_mitbih_local(path, seconds)))
        print(f"DATA: MIT-BIH via GitHub mirror (REAL, {len(data)} record only: {MIRROR_RECORDS}), "
              f"{seconds}s @ {FS} Hz")
        return data, "MIT-BIH (GitHub mirror, 1 record)"
    except Exception as e:
        print(f"[data] GitHub mirror also unavailable ({type(e).__name__}); using SYNTHETIC ECG")
    data = [(f"syn{i}", *synth_ecg(seconds, seed=i)) for i in range(len(RECORDS))]
    print(f"DATA: SYNTHETIC, {len(data)} records x {seconds}s @ {FS} Hz")
    return data, "SYNTHETIC"

def step(b):
    return 1 << (RES - b)


def q_plain(x, b):
    return np.clip(np.floor(x / step(b) + 0.5), 0, (1 << b) - 1).astype(np.int64)


def q_paper(x, b):
    S, lim = step(b), 1 << RES
    out = np.empty(len(x), np.int64)
    cum = 0
    for i, y in enumerate(x.tolist()):
        yq = (y // S) * S
        cum += y - yq
        if cum > S and yq + S < lim:
            yq += S
            cum -= S
        out[i] = yq // S
    return out


def q_shaped(x, b, order=2, dither=False, seed=0):
    S, hi = step(b), (1 << b) - 1
    rng = np.random.default_rng(seed)
    d = (rng.random(len(x)) + rng.random(len(x)) - 1.0) * S if dither else np.zeros(len(x))
    out = np.empty(len(x), np.int64)
    e1 = e2 = 0.0
    for i, y in enumerate(x.tolist()):
        u = y if order == 0 else (y - e1 if order == 1 else y - 2 * e1 + e2)
        c = int(math.floor((u + d[i]) / S + 0.5))
        c = 0 if c < 0 else (hi if c > hi else c)
        e = max(-S, min(S, c * S - u))          
        e2, e1 = e1, e
        out[i] = c
    return out


def dpcm_encode(xl, b, delta):
    half = 1 << (b - 1)
    prev = float(xl[0])
    xr, codes = [prev], [half]
    for v in xl[1:]:
        c = int(math.floor((v - prev) / delta + 0.5))
        c = -half if c < -half else (half - 1 if c > half - 1 else c)
        prev = min(2047.0, max(0.0, prev + c * delta))
        xr.append(prev)
        codes.append(c + half)
    return np.array(codes, np.int64), np.array(xr)


def q_dpcm(x, b):
    xl, best = x.tolist(), None
    for delta in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        codes, xr = dpcm_encode(xl, b, delta)
        p = prd(x, xr, True)
        if best is None or p < best[0]:
            best = (p, codes, xr, delta)
    return best[1], best[2], best[3]
QUANT = {
    "plain": q_plain,
    "paper Alg.1": q_paper,
    "EF-1 round": partial(q_shaped, order=1),
    "EF-2 round": partial(q_shaped, order=2),
    "dither+plain": partial(q_shaped, order=0, dither=True),
}
METHODS = list(QUANT) + ["DPCM"]

# %% ------------------------------ RECONSTRUCTION ------------------------------
def iir_causal(v):
    zi = signal.lfilter_zi(BUTT_B, BUTT_A) * v[0]      # start in steady state (no DC transient)
    return signal.lfilter(BUTT_B, BUTT_A, v, zi=zi)[0]


def fir_causal_comp(v):
    d = (FIR_TAPS - 1) // 2
    pad = np.concatenate([np.full(FIR_TAPS, v[0]), v, np.full(d, v[-1])])
    y = signal.lfilter(FIR_H, 1.0, pad)
    return y[FIR_TAPS + d: FIR_TAPS + d + len(v)]       # remove exact group delay d


def align_lag(x, y, maxlag=40):
    best = (np.inf, 0, y)
    for L in range(maxlag + 1):
        yy = np.concatenate([y[L:], np.full(L, y[-1])]) if L else y
        p = prd(x[:-maxlag], yy[:-maxlag], True)
        if p < best[0]:
            best = (p, L, yy)
    return best[2], best[1]


def decode_variants(x, raw):
    raw = raw.astype(float)
    ff = signal.filtfilt(BUTT_B, BUTT_A, raw)
    cr = iir_causal(raw)
    ca, lag = align_lag(x, cr)
    return {"filtfilt": ff, "iir_causal_raw": cr, "iir_causal_aln": ca,
            "fir_delay_comp": fir_causal_comp(raw)}, lag


def resample_linear(values, src_fs, dst_fs, duration=None):
    n = len(values)
    if duration is None:
        duration = (n - 1) / src_fs
    t_src = np.arange(n) / src_fs
    n_dst = int(round(duration * dst_fs)) + 1
    t_dst = np.arange(n_dst) / dst_fs
    return np.interp(t_dst, t_src, values), t_dst


def reconstruct_at_rate(codes, b, rate, n_orig):
    S = step(b)
    if rate == FS:
        raw = codes.astype(float) * S
        return signal.filtfilt(BUTT_B, BUTT_A, raw), len(codes)
    duration = (len(codes) - 1) / FS
    down, t_down = resample_linear(codes.astype(float), FS, rate, duration)
    t_orig = np.arange(n_orig) / FS
    up = np.interp(t_orig, t_down, down)
    raw = up * S
    return signal.filtfilt(BUTT_B, BUTT_A, raw), len(down)

def prd(x, y, remove_dc=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if remove_dc:
        m = x.mean()
        x, y = x - m, y - m
    return 100 * math.sqrt(np.sum((x - y) ** 2) / np.sum(x ** 2))


def entropy_bits(codes):
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def pack64(codes, b):
    per = 64 // b
    pad = (-len(codes)) % per
    c = np.concatenate([codes, np.zeros(pad, np.int64)]).astype(np.uint64).reshape(-1, per)
    return np.bitwise_or.reduce(c << (np.arange(per, dtype=np.uint64) * np.uint64(b)), axis=1)


def unpack64(words, b, n):
    per = 64 // b
    sh = np.arange(per, dtype=np.uint64) * np.uint64(b)
    return ((words[:, None] >> sh) & np.uint64((1 << b) - 1)).reshape(-1)[:n].astype(np.int64)


def zip_size(words):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        z.writestr("r.bin", words.tobytes())
    return buf.tell()


def simple_qrs(sig):
    b, a = signal.butter(2, [5, 15], btype="band", fs=FS)
    f = signal.filtfilt(b, a, sig - np.mean(sig))
    sq = np.gradient(f) ** 2
    w = int(0.15 * FS)
    ma = np.convolve(sq, np.ones(w) / w, mode="same")
    pk, _ = signal.find_peaks(ma, height=0.25 * np.percentile(ma, 99.5), distance=int(0.25 * FS))
    r = int(0.1 * FS)
    return np.array([max(0, p - r) + np.argmax(np.abs(f[max(0, p - r): p + r])) for p in pk])


def detect_qrs(sig):
    try:
        import wfdb.processing as wp
        d = np.asarray(wp.xqrs_detect(sig=np.asarray(sig, float), fs=FS, verbose=False))
        if len(d) > 3:
            return d
    except Exception:
        pass
    return simple_qrs(np.asarray(sig, float))


def match(ref, test, tol=int(0.1 * FS)):
    ref, test = np.sort(ref), np.sort(test)
    used = np.zeros(len(test), bool)
    tp = 0
    for r in ref:
        j = np.searchsorted(test, r)
        c = [k for k in (j - 1, j) if 0 <= k < len(test) and not used[k] and abs(test[k] - r) <= tol]
        if c:
            used[min(c, key=lambda k: abs(test[k] - r))] = True
            tp += 1
    fn, fp = len(ref) - tp, len(test) - tp
    return 100 * tp / max(tp + fn, 1), 100 * tp / max(tp + fp, 1)


def hrv_err(d0, d1, seconds):
    errs = []
    for m in range(int(seconds // 60)):
        lo, hi = m * 60 * FS, (m + 1) * 60 * FS
        a, c = d0[(d0 >= lo) & (d0 < hi)], d1[(d1 >= lo) & (d1 < hi)]
        if len(a) > 2 and len(c) > 2:
            errs.append(abs(np.diff(a).mean() - np.diff(c).mean()))
    return float(np.mean(errs)) if errs else np.nan

def encode_decode(x, b, method):
    if method == "DPCM":
        codes, xr, delta = q_dpcm(x, b)
        return codes, {"direct": xr}, f"delta={delta}", None
    codes = QUANT[method](x, b)
    variants, lag = decode_variants(x, codes * step(b))
    return codes, variants, f"lag={lag}", codes * step(b)

def run_sweep(data):
    rows, keep = [], {}
    for name, x, beats in data:
        n = len(x)
        for b in BITS_SWEEP:
            for m in METHODS:
                codes, variants, note, raw = encode_decode(x, b, m)
                words = pack64(codes, b)
                assert np.array_equal(unpack64(words, b, n), codes), "pack/unpack mismatch"
                packed_bytes = words.nbytes
                for v, y in variants.items():
                    p_raw, p_dc = prd(x, y), prd(x, y, True)
                    rows.append(dict(
                        record=name, bits=b, method=m, variant=v, note=note,
                        PRD_raw=p_raw, PRD_dc=p_dc,
                        BCR_vs16=2 * n / packed_bytes, BCR_vs11=(RES * n / 8) / packed_bytes,
                        H_bits=entropy_bits(codes),
                        save_vs16=100 * (1 - packed_bytes / (2 * n)),
                        save_zip=100 * (1 - zip_size(words) / (2 * n)),
                        QS_raw=(2 * n / packed_bytes) / max(p_raw, 1e-6),
                        QS_dc=(2 * n / packed_bytes) / max(p_dc, 1e-6)))
                if name == data[0][0] and b == BITS_MAIN:
                    keep[m] = dict(codes=codes, variants=variants, raw=raw)
        print(f"  swept record {name}")
    return pd.DataFrame(rows), keep


def run_qrs(data):
    rows = []
    for name, x, beats in data:
        if name in PACEMAKER_RECORDS:
            continue    # paper excludes 102, 104, 107, 217 here (pacemaker signals confuse R-peak detectors)
        secs = len(x) / FS
        d_orig = detect_qrs(x)
        se, pp = match(beats, d_orig)
        rows.append(dict(record=name, method="ORIGINAL 11-bit", variant="-", SE=se, PP=pp, HV_samples=0.0))
        for m in METHODS:
            _, variants, _, _ = encode_decode(x, BITS_MAIN, m)
            for v in (["direct"] if m == "DPCM" else ["filtfilt", "fir_delay_comp"]):
                d = detect_qrs(variants[v])
                se, pp = match(beats, d)
                rows.append(dict(record=name, method=m, variant=v, SE=se, PP=pp,
                                 HV_samples=hrv_err(d_orig, d, secs)))
    return pd.DataFrame(rows)


def run_resample_sweep(data):
    rows = []
    for name, x, beats in data:
        for b in BITS_SWEEP:
            codes = q_paper(x, b)
            for rate in RESAMPLE_RATES:
                y, n_stored = reconstruct_at_rate(codes, b, rate, len(x))
                p_raw = prd(x, y)                       # raw PRD only -- matches the paper's
                bcr = 16 * FS / (b * rate)               # own numbers (below ~5%), avoids mixing
                rows.append(dict(record=name, bits=b, rate=rate,   # in the huge DC-removed values
                                  PRD_raw=p_raw, BCR_vs16=bcr, n_stored=n_stored))
        print(f"  resample-swept record {name}")
    return pd.DataFrame(rows)


def run_qrs_vs_bits(data):
    rows = []
    for name, x, beats in data:
        if name in PACEMAKER_RECORDS:
            continue
        for b in BITS_SWEEP:
            codes = q_paper(x, b)
            for rate in RESAMPLE_RATES:
                y, _ = reconstruct_at_rate(codes, b, rate, len(x))
                d = simple_qrs(y)
                se, pp = match(beats, d)
                rows.append(dict(record=name, bits=b, rate=rate, SE=se, PP=pp))
        print(f"  qrs-vs-bits record {name}")
    return pd.DataFrame(rows)


def paper_claim_check():
    n = 48 * 650000
    comp = n * 4 / 8 / 2 ** 20
    o16 = n * 2 / 2 ** 20
    print("\n--- arithmetic check of the paper's 90.4% ---")
    print(f"48 records x 650000 samples @4 bit = {comp:.1f} MiB   (paper Table 3: 15.0 MB)")
    print(f"vs one 16-bit channel ({o16:.1f} MiB): {100 * (1 - comp / o16):.1f}% saving  (= 1 - 4/16)")
    print(f"vs paper's 155 MB baseline: {100 * (1 - 15.0 / 155):.1f}%  (baseline is {155 / o16:.1f}x the 16-bit channel)")

def make_plots(df, keep, data):
    name, x, _ = data[0]
    sel = df[df.variant.isin(["filtfilt", "direct"])]
    g = sel.groupby(["method", "bits"])[["PRD_raw", "PRD_dc"]].mean().reset_index()
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

    # 3) time-domain overlay
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
    name, x, _ = data[0]
    i0 = 3 * FS
    sl = slice(i0, i0 + 250)  
    t = np.arange(len(x))[sl]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, b in zip(axes.flat, HIGHLIGHT_BITS):
        codes = q_paper(x, b)
        y, _ = reconstruct_at_rate(codes, b, FS, len(x))
        ax.plot(t, x[sl], "b", lw=1, label="Original")
        ax.plot(t, y[sl], "r", lw=1, label="Reconstructed")
        ax.set_title(f"{b}-bit")
        ax.set_xlabel("Sample points")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7)
    fig.suptitle(f"Original vs reconstructed at different bit depths (record {name}, paper Alg.1)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_reconstruction_grid.png", dpi=130)


def fig_prd_bcr_vs_bits(resample_df):
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
    """Paper's Fig. 6: (a) sensitivity, (b) +P vs number of bits, one line per sample rate,
    Pan-Tompkins-style detector, paper's algorithm only."""
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

if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    data, src = get_data(DURATION_S)
    df, keep = run_sweep(data)
    df.to_csv(f"{OUT_DIR}/sweep_all.csv", index=False)

    main = df[df.bits == BITS_MAIN].groupby(["method", "variant"]).agg(
        PRD_raw=("PRD_raw", "mean"), PRD_dc=("PRD_dc", "mean"),
        BCR_vs16=("BCR_vs16", "mean"), BCR_vs11=("BCR_vs11", "mean"), H_bits=("H_bits", "mean"),
        save_vs16=("save_vs16", "mean"), save_zip=("save_zip", "mean"),
        QS_raw=("QS_raw", "mean"), QS_dc=("QS_dc", "mean")).reset_index()
    print(f"\n=== {BITS_MAIN}-bit results, MEAN-OF-PER-RECORD-PRD over {len(data)} records ({src}) ===")
    print(main.sort_values("PRD_dc").to_string(index=False))
    main.to_csv(f"{OUT_DIR}/table_main.csv", index=False)
    pool_rows = []
    for m in METHODS:
        for v in (["direct"] if m == "DPCM" else ["filtfilt", "iir_causal_aln", "fir_delay_comp"]):
            se = sse = 0.0
            se_dc = sse_dc = 0.0
            for name, x, _ in data:
                codes, variants, _, _ = encode_decode(x, BITS_MAIN, m)
                if v not in variants:
                    continue
                y = variants[v]
                xf, yf = np.asarray(x, float), np.asarray(y, float)
                se += np.sum((xf - yf) ** 2); sse += np.sum(xf ** 2)
                xd, yd = xf - xf.mean(), yf - yf.mean()
                se_dc += np.sum((xd - yd) ** 2); sse_dc += np.sum(xd ** 2)
            pool_rows.append(dict(method=m, variant=v,
                                   PRD_pooled_raw=100 * math.sqrt(se / sse),
                                   PRD_pooled_dc=100 * math.sqrt(se_dc / sse_dc)))
    pool_df = pd.DataFrame(pool_rows)
    print(f"\n=== {BITS_MAIN}-bit results, POOLED PRD (one PRD over all {len(data)} concatenated records) ===")
    cmp = main.merge(pool_df, on=["method", "variant"], how="inner")
    print(cmp[["method", "variant", "PRD_raw", "PRD_pooled_raw", "PRD_dc", "PRD_pooled_dc"]]
          .sort_values("PRD_pooled_dc").to_string(index=False))
    cmp.to_csv(f"{OUT_DIR}/table_prd_pooled_vs_mean.csv", index=False)

    print("\n=== QRS detection + HRV at", BITS_MAIN, "bits ===")
    q = run_qrs(data).groupby(["method", "variant"])[["SE", "PP", "HV_samples"]].mean().reset_index()
    print(q.to_string(index=False))
    q.to_csv(f"{OUT_DIR}/table_qrs.csv", index=False)

    print(f"\n=== paper's Fig. 5 reproduction: PRD & BCR vs bits, {DURATION_S}s/record, "
          f"paper Alg.1 only, rates {RESAMPLE_RATES} ===")
    if DURATION_S == DURATION_S:
        data_full, src_full = data, src        # avoid re-downloading the same thing twice
    else:
        print(f"  loading full-length ({DURATION_S}s) data for the paper-reproduction "
              f"figures -- this is a second, separate download/load from the {DURATION_S}s set above")
        data_full, src_full = get_data(DURATION_S)
    resample_df = run_resample_sweep(data_full)
    resample_df.to_csv(f"{OUT_DIR}/table_prd_bcr_vs_bits.csv", index=False)
    print(resample_df.groupby(["rate", "bits"])[["PRD_raw", "BCR_vs16"]].mean().reset_index().to_string(index=False))
    fig_prd_bcr_vs_bits(resample_df)

    print(f"\n=== paper's Fig. 6 reproduction: QRS SE/+P vs bits, {DURATION_S}s/record, "
          f"paper Alg.1 only, rates {RESAMPLE_RATES} (this is the slow one) ===")
    qrsbits_df = run_qrs_vs_bits(data_full)
    qrsbits_df.to_csv(f"{OUT_DIR}/table_qrs_vs_bits.csv", index=False)
    print(qrsbits_df.groupby(["rate", "bits"])[["SE", "PP"]].mean().reset_index().to_string(index=False))
    fig_qrs_vs_bits(qrsbits_df)

    fig_reconstruction_grid(data_full)

    paper_claim_check()
    make_plots(df, keep, data)
    print(f"\nSaved tables + figures in ./{OUT_DIR}/")
