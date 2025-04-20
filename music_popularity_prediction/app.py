import streamlit as st
import numpy as np
import joblib

# Set page config first
st.set_page_config(page_title="Music Popularity Predictor", page_icon="🎧", layout="wide")

# Load pre-trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Header
st.title("🎧 Music Popularity Predictor")
st.markdown("Use audio features to predict how popular a music track could be! 🔥")

st.image("https://media.giphy.com/media/26AHONQ79FdWZhAI0/giphy.gif", use_column_width=True)

# Sidebar inputs
with st.sidebar:
    st.header("🎛️ Input Audio Features")
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -5.0)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.2)

# Main button and prediction
if st.button("🎯 Predict Popularity"):
    features = np.array([[energy, valence, danceability, loudness,
                          acousticness, tempo, speechiness, liveness]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("🎵 Popularity Score", f"{prediction[0]:.2f}")
    with col2:
        st.success("✅ Prediction successful! This could be a banger! 💥")

# Feature explanation
with st.expander("ℹ️ Feature Descriptions"):
    st.markdown("""
    - **Energy**: Perceptual measure of intensity and activity.
    - **Valence**: Musical positiveness (0 = sad, 1 = happy).
    - **Danceability**: How suitable the track is for dancing.
    - **Loudness**: Overall loudness in decibels (dB).
    - **Acousticness**: Confidence measure of whether the track is acoustic.
    - **Tempo**: Estimated tempo in beats per minute (BPM).
    - **Speechiness**: Presence of spoken words.
    - **Liveness**: Detects the presence of an audience in the recording.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ by <b>Hariprasath S.</b> | Follow me on <a href='https://github.com/yourusername' target='_blank'>GitHub</a></center>",
    unsafe_allow_html=True
)
