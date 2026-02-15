import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Voice Detector", layout="wide")
st.title("🎙️ Audio Feature Analysis")
st.write("Upload an audio file to analyze its characteristics.")

uploaded_file = st.sidebar.file_uploader("Upload audio (wav/mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('Processing...'):
        y, sr = librosa.load(uploaded_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{duration:.2f} s")
        col2.metric("Sample Rate", f"{sr} Hz")
        
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        st.divider()
        st.subheader("Analysis Result")
        if rms < 0.02 and zcr < 0.02:
            st.error("⚠️ Likely AI Voice (Smooth fluctuations)")
        else:
            st.success("✅ Likely Human Voice (Natural fluctuations)")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.set_title("Audio Waveform")
        ax1.plot(y, color='dodgerblue')
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax2)
        fig.colorbar(img, ax=ax2)
        ax2.set_title("MFCC Spectrogram")
        plt.tight_layout()
        st.pyplot(fig)
