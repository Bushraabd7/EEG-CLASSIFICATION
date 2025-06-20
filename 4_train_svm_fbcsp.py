import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# 📥 Load separated FBCSP features
# ---------------------------
train_data = np.load("features_fbcsp_train.npz")
test_data = np.load("features_fbcsp_test.npz")

X_train = train_data["X"]
y_train = train_data["y"]
X_test = test_data["X"]
y_test = test_data["y"]

# ---------------------------
# ⚙️ Feature normalization
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 🧠 Train SVM on Train CSP features
# ---------------------------
clf = SVC(kernel='linear', C=1, class_weight='balanced', random_state=42)
clf.fit(X_train_scaled, y_train)

# ---------------------------
# 🧪 Predict only on first 400 test samples
# ---------------------------
X_test_400 = X_test_scaled[:400]
y_test_400 = y_test[:400]
y_pred = clf.predict(X_test_400)

# ---------------------------
# 🎯 Evaluation
# ---------------------------
acc = accuracy_score(y_test_400, y_pred)
print(f"SVM Test Accuracy: {acc * 100:.2f}%")
print("\n📋 Classification Report:\n", classification_report(y_test_400, y_pred))

# 🔍 Confusion Matrix
cm = confusion_matrix(y_test_400, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2], yticklabels=[1, 2])
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
