import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import scipy.signal  # For Wiener filtering [cite: 58]
import matplotlib.pyplot as plt # For visual feedback [cite: 60]

# --- CUSTOM STYLING (CSS) ---
def style_app():
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }

        .block-container {
            padding-top: 3rem;
        }
        
        # Inside style_app()
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5rem;
    
            text-shadow: 0px 4px 15px rgba(255, 215, 0, 0.3); 
        }

        .sub-text {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2rem;
        }

        .stFileUploader section {
            background-color: #1e293b !important;
            border: 2px dashed #334155 !important;
            border-radius: 15px !important;
        }

        .stAlert {
            border-radius: 10px !important;
            border: none !important;
        }

        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            border-radius: 50px;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0px 4px 15px rgba(58, 123, 213, 0.4);
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0px 6px 20px rgba(58, 123, 213, 0.6);
            color: #ffffff;
        }

        audio {
            width: 100%;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- UPDATED PROCESSING LOGIC ---
def process_audio(input_audio, sensitivity):
    y, sr = librosa.load(input_audio, sr=None)
    stft = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft)

    # NEW: Detect noise irrespective of timing
    # Instead of first 15 frames, we find the median/minimum 
    # magnitude across the whole clip to identify persistent noise.
    # This 'Rolling' approach judges noise even if it moves.
    noise_floor = np.median(magnitude, axis=1, keepdims=True)
    
    # Apply spectral subtraction using the global noise floor
    # Capped at your requested 3.0 limit to prevent 'shredding'
    clean_mag = np.maximum(magnitude - (noise_floor * sensitivity), 0)
    
    # Keep the Wiener smoothing you liked for that clean finish
    clean_mag = scipy.signal.wiener(clean_mag)

    y_clean = librosa.istft(clean_mag * phase)

    buffer = io.BytesIO()
    sf.write(buffer, y_clean, sr, format='WAV')
    buffer.seek(0)
    return buffer, y, y_clean, sr

# --- APP LAYOUT ---
st.set_page_config(page_title="NOICE", page_icon="✨", layout="centered")
style_app()

# Sidebar for Pro Controls [cite: 34, 59]
st.sidebar.header(" Algorithm Setting ")
reduction_strength = st.sidebar.slider(
    "Reduction Strength", 
    min_value=1.0, 
    max_value=3.0,  # Capped at 3.0 to prevent "shredding" 
    value=1.5, 
    step=0.1,
    help="Higher values (up to 3.0) remove more noise but may distort the voice."
)

# Locate this line and replace it:
st.markdown(
    '<h1 class="main-header">'
    '<span style="color: #FFD700;">NOI</span>'
    '<span style="color: #DAA520;">CE</span>'
    '</h1>', 
    unsafe_allow_html=True
)
st.markdown('<p class="sub-text">Professional Grade Fourier-Transform Noise Reduction</p>', unsafe_allow_html=True)

with st.container():
    uploaded_file = st.file_uploader("", type=["wav", "mp3"])

if uploaded_file is not None:
    st.info("🎵 File Loaded Successfully")
    st.audio(uploaded_file)

    if st.button("🚀 CLEAN AUDIO NOW"):
        with st.spinner("Analyzing spectral baseline and filtering..."):
            # Process and get raw arrays for visualization [cite: 57, 61]
            processed_data, original_y, clean_y, sr = process_audio(uploaded_file, reduction_strength)

            st.success("✨ Audio Restored!")

            # Visual Waveform Comparison [cite: 31, 32, 60]
            st.markdown("### 📊 Waveform Comparison")
            fig, ax = plt.subplots(2, 1, figsize=(10, 5))
            plt.subplots_adjust(hspace=0.5)
            
            # Original Waveform
            librosa.display.waveshow(original_y, sr=sr, ax=ax[0], color='gray', alpha=0.5)
            ax[0].set_title("Original (Dirty)", color="#94a3b8")
            
            # Cleaned Waveform
            librosa.display.waveshow(clean_y, sr=sr, ax=ax[1], color='#00d4ff')
            ax[1].set_title("Processed (Clean)", color="#00d4ff")
            
            st.pyplot(fig)

            st.markdown("### 🎧 Result")
            st.audio(processed_data)

            st.download_button(
                label="📥 DOWNLOAD CLEANED WAV",
                data=processed_data,
                file_name="sonic_clean_output.wav",
                mime="audio/wav"
            )

st.markdown("---")


