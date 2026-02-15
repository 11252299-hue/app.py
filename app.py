import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI語音偵測", layout="centered")
st.title("🎙️ 音訊分析工具")

uploaded_file = st.file_uploader("選擇音訊檔案 (wav/mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        with st.spinner('正在讀取音訊...'):
            # 讀取音訊，加入例外處理與重置檔案指標
            uploaded_file.seek(0)
            y, sr = librosa.load(uploaded_file, sr=None)
            
            st.success("✅ 檔案讀取成功！")
            
            # 計算特徵
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
            
            col1, col2 = st.columns(2)
            col1.metric("平均能量 (RMS)", f"{rms:.4f}")
            col2.metric("過零率 (ZCR)", f"{zcr:.4f}")
            
            if rms < 0.02 and zcr < 0.02:
                st.warning("⚠️ 可能是 AI 語音（波動平滑）")
            else:
                st.info("✅ 可能是真人語音（自然波動）")
            
            # 畫波形
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(y, color='#1f77b4')
            ax.set_title("Waveform")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"讀取失敗：您的檔案格式可能不正確。")
        st.write(f"錯誤詳情: {e}")
