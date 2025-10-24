import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

DATA_DIR = 'data'
X, y = [], []

labels = sorted(os.listdir(DATA_DIR))
for label in labels:
    folder_path = os.path.join(DATA_DIR, label)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            data = np.loadtxt(file_path)
            data = data.reshape(-1, 3)
            if data.shape[0] != 21:
                print(f"⚠️ Skipped {file_path} due to missing landmarks")
                continue

            data -= data[0]  # normalize by wrist
            X.append(data.flatten())
            y.append(label)
        except:
            print(f"⚠️ Error loading {file_path}")

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Model accuracy: {accuracy * 100:.2f}%")

with open("sign_model.pkl", "wb") as f:
    pickle.dump((model, le), f)

print("✅ Model and LabelEncoder saved as 'sign_model.pkl'")
print("Labels:", le.classes_)
