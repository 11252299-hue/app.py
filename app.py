import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt

st.title("🎙️ 音訊分析工具 (修復版)")

uploaded_file = st.file_uploader("上傳音訊", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        # 讀取音訊
        y, sr = librosa.load(uploaded_file, sr=None)
        st.success("檔案讀取成功！")
        
        # 顯示波形
        fig, ax = plt.subplots()
        ax.plot(y)
        st.pyplot(fig)
        
        # 計算簡單特徵
        rms = np.mean(librosa.feature.rms(y=y))
        st.write(f"平均能量 (RMS): {rms:.4f}")
        
    except Exception as e:
        st.error(f"發生錯誤: {e}")
