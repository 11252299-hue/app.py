import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Voice Detector", layout="wide")
st.title("🎙️ Audio Analysis Tool")

uploaded_file = st.sidebar.file_uploader("Upload audio (wav/mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('Processing...'):
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 計算特徵
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        st.subheader("Analysis Result")
        if rms < 0.02 and zcr < 0.02:
            st.error("⚠️ Likely AI Voice")
        else:
            st.success("✅ Likely Human Voice")
            
        fig, ax = plt.subplots()
        ax.plot(y)
        st.pyplot(fig)
