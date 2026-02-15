import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# 網頁標題與設定
st.set_page_config(page_title="AI 語音偵測器", layout="wide")
st.title("🎙️ 語音特徵分析工具")
st.write("上傳音訊檔案，讓我們分析它是 AI 還是真人聲音。")

# 側邊欄：上傳檔案
uploaded_file = st.sidebar.file_uploader("請上傳音訊檔 (wav / mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('正在處理音訊...'):
        # 1. 讀取音訊
        y, sr = librosa.load(uploaded_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. 顯示基本資訊
        col1, col2, col3 = st.columns(3)
        col1.metric("音訊長度", f"{duration:.2f} 秒")
        col2.metric("取樣率", f"{sr} Hz")
        
        # 3. 計算特徵
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        # 4. 判斷邏輯
        st.divider()
        st.subheader("分析結果")
        if rms < 0.02 and zcr < 0.02:
            st.error("⚠️ 可能是 AI 語音（波動過於平滑）")
        else:
            st.success("✅ 可能是真人語音（具備自然波動）")

        # 5. 繪製圖表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 波形圖
        ax1.set_title("Audio Waveform")
        ax1.plot(y, color='dodgerblue')
        
        # MFCC
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax2)
        fig.colorbar(img, ax=ax2)
        ax2.set_title("MFCC Spectrogram")
        
        plt.tight_layout()
        st.pyplot(fig)

        # 顯示數值細節
        with st.expander("查看原始特徵數據"):
            st.write(f"RMS: {rms:.5f}")
            st.write(f"ZCR: {zcr:.5f}")
            st.write("MFCC Means:", mfcc_mean)
else:
    st.info("請從左側上傳檔案以開始分析。")
