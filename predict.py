import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import kurtosis, skew, iqr

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

# ---------------- FEATURE EXTRACTION (YOUR 8 FEATURES) ----------------
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

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

def predict_from_df(df):
    x = df["eda"].values
    feats = extract_features(x).reshape(1, -1)
    return model.predict(feats)[0]
