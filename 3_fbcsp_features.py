import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import train_test_split

# ---------------------------
# 📥 Load filtered bands
# ---------------------------
data = np.load("eeg_fbcsp_all_bands.npz", allow_pickle=True)
X_bands = data["X_bands"]  # قائمة من 6 نطاقات (trials, channels, timepoints)
y = data["y"]
subjects = data["subjects"]

# ---------------------------
# 🔀 Split data (global split)
# ---------------------------
X_bands = [np.array(X) for X in X_bands]  # تأكد من التحويل من list إلى np.array
X_train_bands, X_test_bands = [], []
y_train, y_test, subjects_train, subjects_test = train_test_split(
    y, subjects, test_size=0.2, stratify=y, random_state=42
)

train_idx = np.isin(subjects, subjects_train)
test_idx = np.isin(subjects, subjects_test)

for X in X_bands:
    X_train_bands.append(X[train_idx])
    X_test_bands.append(X[test_idx])

# ---------------------------
# 🧠 Apply CSP per band and collect features
# ---------------------------
X_train_all = []
X_test_all = []

for i in range(len(X_bands)):
    print(f"🔍 Applying CSP for Band {i + 1}")
    csp = CSP(n_components=30, reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train_bands[i], y[train_idx])
    X_test_csp = csp.transform(X_test_bands[i])

    X_train_all.append(X_train_csp)
    X_test_all.append(X_test_csp)

# ---------------------------
# 🔗 Concatenate features from all bands
# ---------------------------
X_train_final = np.concatenate(X_train_all, axis=1)  # shape: (n_trials, 2*num_bands)
X_test_final = np.concatenate(X_test_all, axis=1)

print(f"✅ FBCSP Feature Shapes → Train: {X_train_final.shape}, Test: {X_test_final.shape}")

# 💾 Save features separately
# ---------------------------
np.savez("features_fbcsp_train.npz",
         X=X_train_final,
         y=y[train_idx])

np.savez("features_fbcsp_test.npz",
         X=X_test_final,
         y=y[test_idx])

print("💾 Saved as 'features_fbcsp_train.npz' and 'features_fbcsp_test.npz'")

