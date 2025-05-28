import numpy as np
from scipy.signal import butter, filtfilt

# ---------------------------
# 📥 Load raw EEG data
# ---------------------------
data = np.load("all_raw_eeg.npz", allow_pickle=True)
X_raw = data["X"]          # shape: (trials, channels, timepoints)
y = data["y"]
subjects = data["subjects"]

# ---------------------------
# ⚙️ Settings
# ---------------------------
fs = 514
start_sec = 2
end_sec = 5
start_sample = int(start_sec * fs)
end_sample = int(end_sec * fs)

# 🧠 Filter bank frequency ranges
filter_bands = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 30)
]

# ---------------------------
# 🎚️ Bandpass filter functions
# ---------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_filter(X, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    X_filt = np.zeros_like(X)
    for trial in range(X.shape[0]):
        for ch in range(X.shape[1]):
            X_filt[trial, ch, :] = filtfilt(b, a, X[trial, ch, :])
    return X_filt

# ---------------------------
# 🔄 Apply filter bank
# ---------------------------
X_all_bands = []

for i, (low, high) in enumerate(filter_bands):
    print(f"🔄 Filtering band {i+1}: {low}-{high} Hz")

    X_filt = apply_filter(X_raw, low, high, fs)
    X_trimmed = X_filt[:, :, start_sample:end_sample]

    # ⚖️ Normalize per trial
    for t in range(X_trimmed.shape[0]):
        for ch in range(X_trimmed.shape[1]):
            signal = X_trimmed[t, ch]
            X_trimmed[t, ch] = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

    X_all_bands.append(X_trimmed)

print("✅ Filter bank complete. Total bands:", len(X_all_bands))

# ---------------------------
#  Save all bands
# ---------------------------
np.savez("eeg_fbcsp_all_bands.npz", X_bands=X_all_bands, y=y, subjects=subjects)
print("💾 Saved as 'eeg_fbcsp_all_bands.npz'")
