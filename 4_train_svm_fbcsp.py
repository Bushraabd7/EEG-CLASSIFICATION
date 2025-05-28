import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# 📥 Load all features (merged FBCSP)
# ---------------------------
data = np.load("features_fbcsp.npz")
X_all = np.concatenate([data["X_train"], data["X_test"]], axis=0)
y_all = np.concatenate([data["y_train"], data["y_test"]], axis=0)

# ---------------------------
# 🔀 Re-split into Train/Test
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

# ---------------------------
# ⚙️ Feature normalization
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 🧠 Train SVM
# ---------------------------
clf = SVC(kernel='linear', C=1, class_weight='balanced', random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# ---------------------------
# 🎯 Evaluation
# ---------------------------
acc = accuracy_score(y_test, y_pred)
print(f"✅ SVM Test Accuracy (New Split): {acc * 100:.2f}%")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 🔍 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2], yticklabels=[1, 2])
plt.title("Confusion Matrix (SVM - New Split)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()