import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 載入模型
with open("random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# 滑動視窗編碼
def sliding_window_encode(seq, window_size=5):
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    padding = window_size // 2
    padded_seq = ["X"] * padding + list(seq) + ["X"] * padding
    features = []

    for i in range(len(seq)):
        window = padded_seq[i:i + window_size]
        feature = [aa_to_index.get(aa, 20) for aa in window]
        features.append(feature)

    return np.array(features)

# Streamlit 介面
st.title("🔬 ATP Binding Site Predictor")
st.markdown("輸入蛋白質序列，我們將預測其中哪些位置可能是 ATP 結合位點。")

sequence_input = st.text_area("請輸入蛋白質序列（只接受一行純字母）", height=150)

if sequence_input:
    sequence_input = sequence_input.strip().upper()

    if not sequence_input.isalpha():
        st.error("❌ 請只輸入英文字母（蛋白質序列）")
    elif len(sequence_input) < 5:
        st.warning("⚠️ 序列長度過短，至少需要 5 個氨基酸")
    else:
        X = sliding_window_encode(sequence_input)
        probs = model.predict_proba(X)[:, 1]
        threshold = 0.2
        predicted_labels = (probs >= threshold).astype(int)
        indices = np.where(predicted_labels == 1)[0]

        # 顯示機率範圍
        st.markdown(f"🔍 **預測機率範圍**：最小 = {probs.min():.3f}，最大 = {probs.max():.3f}")

        # 顯示圖形
        st.markdown("📈 **機率折線圖**（預測結合位點機率）")
        fig, ax = plt.subplots()
        ax.plot(probs, label="Binding probability", color="blue")
        ax.axhline(y=threshold, color="red", linestyle="--", label=f"閾值 = {threshold}")
        ax.set_xlabel("位置")
        ax.set_ylabel("機率")
        ax.legend()
        st.pyplot(fig)

        if len(indices) == 0:
            st.warning("⚠️ 沒有預測出任何結合位點（機率未超過 0.2）")
        else:
            st.success(f"✅ 預測出 {len(indices)} 個可能的結合位點：")
            st.markdown(", ".join(str(i) for i in indices))
