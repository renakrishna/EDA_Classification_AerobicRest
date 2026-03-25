import numpy as np
import joblib
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import kurtosis, skew, iqr

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- FILTERS ----------------
def _filtfilt_safe(b, a, x):
    x = np.asarray(x, dtype=float).squeeze()
    return filtfilt(b, a, x)

def butter_bandpass(x, fs, low=0.01, high=1.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return _filtfilt_safe(b, a, x)

def butter_lowpass(x, fs, cutoff=0.05, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype="low")
    return _filtfilt_safe(b, a, x)

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(x, fs=4.0):
    x = np.asarray(x, dtype=float).squeeze()

    tonic = butter_lowpass(x, fs, 0.05)
    phasic = butter_bandpass(x, fs, 0.1, 0.5)

    peaks, _ = find_peaks(phasic, prominence=np.std(phasic))
    peak_amps = phasic[peaks] if len(peaks) else np.array([0.0])

    duration_sec = len(x) / fs
    scr_peaks_per_min = len(peaks) / ((duration_sec/60) + 1e-8)

    freqs, psd = welch(x, fs=fs)
    psd_peaks, _ = find_peaks(psd)
    psd_peak_amps = psd[psd_peaks] if len(psd_peaks) else np.array([0.0])

    return np.array([
        np.mean(tonic),
        np.mean(phasic) if len(phasic) else 0,
        scr_peaks_per_min,
        kurtosis(x),
        skew(x),
        np.max(peak_amps),
        iqr(x),
        np.max(psd_peak_amps)
    ])

# ---------------- MAIN PREDICTION ----------------
def predict_full_signal(df, fs=4.0):
    x = df["eda"].values

    # Segment positions (from your experiment)
    rest_end = int(120 * fs)
    aero_start = int(1560 * fs)
    aero_end = int(1740 * fs)

    if len(x) < aero_end:
        raise ValueError("Signal too short (needs ~1740 seconds)")

    rest = x[:rest_end]
    aerobic = x[aero_start:aero_end]

    rest_feat = extract_features(rest, fs).reshape(1, -1)
    aero_feat = extract_features(aerobic, fs).reshape(1, -1)

    rest_pred = model.predict(rest_feat)[0]
    aero_pred = model.predict(aero_feat)[0]

    rest_prob = model.predict_proba(rest_feat)[0]
    aero_prob = model.predict_proba(aero_feat)[0]

    return {
        "rest_pred": rest_pred,
        "aero_pred": aero_pred,
        "rest_conf": max(rest_prob),
        "aero_conf": max(aero_prob)
    }
