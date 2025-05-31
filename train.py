import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier

# 滑動視窗編碼
def sliding_window_encode(seq, window_size=5):
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    padding = window_size // 2
    padded = ['X'] * padding + list(seq) + ['X'] * padding
    features = []

    for i in range(len(seq)):
        window = padded[i:i+window_size]
        feature = []
        for aa in window:
            if aa in AMINO_ACIDS:
                feature.append(AMINO_ACIDS.index(aa))
            else:
                feature.append(20)
        features.append(feature)

    return np.array(features)

# 讀取資料
with open("converted_dataset.json", "r") as f:
    dataset = json.load(f)

X_all = []
y_all = []

for entry in dataset:
    seq = entry["sequence"]
    labels = entry["labels"]
    X = sliding_window_encode(seq, window_size=5)
    y = np.array(labels)
    X_all.append(X)
    y_all.append(y)

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

# 印出資料分析資訊
print(f"✅ 總樣本數：{len(y_all)}, 特徵維度：{X_all.shape[1]}")
print(f"✅ 正樣本（binding site=1）：{np.sum(y_all)}")
print(f"✅ 負樣本（非binding site=0）：{len(y_all) - np.sum(y_all)}")

# 訓練模型
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_all, y_all)

# 儲存模型
with open("random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ 模型訓練完成，並已儲存為 random_forest.pkl")
