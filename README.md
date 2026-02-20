import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# 網頁基礎設定
st.set_page_config(page_title="AI 語音防詐偵測系統", layout="centered")
st.title("🛡️ AI 語音與詐騙風險偵測")
st.markdown("---")

# 檔案上傳介面
uploaded_file = st.file_uploader("請上傳音訊檔案進行深度辨識 (wav/mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        with st.spinner('正在分析數位指紋與語音特徵...'):
            # 讀取音訊
            uploaded_file.seek(0)
            y, sr = librosa.load(uploaded_file, sr=None)
            
            # --- 深度特徵提取 ---
            # 1. MFCC (梅爾倒頻譜係數)：分析音色是否具備 AI 的過度平滑感
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_std = np.std(mfccs) 

            # 2. 頻譜質心：觀察聲音的亮度與高頻分佈
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            avg_centroid = np.mean(centroid)

            # 3. 頻譜平坦度：判斷是自然的噪音還是規律的合成聲
            flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            st.success("✅ 分析流程完成")

            # --- 顯示關鍵指標 ---
            col1, col2, col3 = st.columns(3)
            col1.metric("音色穩定度", f"{mfcc_std:.2f}")
            col2.metric("頻譜重心", f"{int(avg_centroid)} Hz")
            col3.metric("數位平坦度", f"{flatness:.4f}")

            # --- 強化版判定邏輯 ---
            # 真人語音的變化度(std)通常較大，且高頻細節較多
            is_ai_risk = False
            risk_score = 15 # 初始基礎分

            if mfcc_std < 45.0:  # AI 特有的平滑指紋
                is_ai_risk = True
                risk_score += 40
            if avg_centroid < 2600: # AI 數位濾波痕跡
                is_ai_risk = True
                risk_score += 30
            
            st.markdown("### 偵測評估報告")
            if is_ai_risk or risk_score > 50:
                st.error(f"🚨 高風險警告：疑似 AI 合成語音 (風險指數: {min(risk_score, 100)}%)")
                st.write("**建議：** 請謹慎對待通話內容，對方可能使用 Deepfake 技術。")
            else:
                st.info(f"✅ 安全：特徵符合真人語音規律 (風險指數: {risk_score}%)")
                st.write("**建議：** 未偵測到明顯 AI 痕跡，但仍需注意通話中的詐騙關鍵字。")

            st.progress(min(risk_score, 100))

            # --- 可視化圖表 (聲譜圖) ---
            st.write("### 聲譜圖 (Spectrogram) 分析")
            st.write("註：AI 生成的聲音在上方高頻處通常過於乾淨或有異常條紋。")
            fig, ax = plt.subplots(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_DB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"分析失敗：請確認檔案格式是否正確。")
        st.write(f"錯誤訊息：{e}")
