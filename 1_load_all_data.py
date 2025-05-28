import os
import numpy as np
import scipy.io

# Define the source directory
sourcedata_dir = r"C:\Users\bushr\Music\eeg project 5\sourcedata"

X_list = []
y_list = []
subjects_list = []

for subject_folder in sorted(os.listdir(sourcedata_dir)):
    subject_path = os.path.join(sourcedata_dir, subject_folder)
    if os.path.isdir(subject_path):
        for file in os.listdir(subject_path):
            if file.endswith(".mat"):
                file_path = os.path.join(subject_path, file)
                print(f"📂 Loading: {file_path}")

                mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
                eeg = mat['eeg']

                X = eeg.rawdata       # shape: (trials, channels, timepoints)
                y = np.array(eeg.label)

                X_list.append(X)
                y_list.append(y)

                subject_id = int(subject_folder.split("-")[1])
                subjects_list.extend([subject_id] * len(y))

# Concatenate all data
X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)
subjects_all = np.array(subjects_list)

print("✅ Final data shape:", X_all.shape)
print("✅ Labels shape:", y_all.shape)
print("✅ Subjects shape:", subjects_all.shape)
print("🧠 Unique classes:", np.unique(y_all))

# Save
np.savez("all_raw_eeg.npz", X=X_all, y=y_all, subjects=subjects_all)
print("💾 Saved as 'all_raw_eeg.npz'")
