import streamlit as st
import numpy as np
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap");

    html, body, [class*="css"] {
        font-family: "Poppins", sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 20% 20%, #1e1b4b 0%, #0f0c29 45%, #050510 100%);
        color: #f1f5f9;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .hero {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 300;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .section-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #a78bfa;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7c3aed, #3b82f6);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.7rem;
        border-radius: 12px;
        border: none;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6);
        color: white;
    }

    .result-number {
        font-size: 6rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        background: linear-gradient(90deg, #34d399, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .confidence-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    .placeholder-box {
        text-align: center;
        padding: 3rem 1rem;
        color: #64748b;
        font-size: 0.95rem;
    }

    canvas {
        border-radius: 16px !important;
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.35);
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

model = keras.models.load_model("mnist_cnn_model.h5")

st.markdown("""
    <div class="hero">
        <h1>Digit Recognition AI</h1>
        <p>Draw any digit from 0 to 9 and watch a Convolutional Neural Network read your handwriting in real time.</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Draw Here</div>", unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=18,
        stroke_color="#ffffff",
        background_color="#0a0a1a",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas"
    )
    predict_clicked = st.button("✨ Predict Digit")
    clear_note = st.caption("Use the canvas toolbar (bottom-right of canvas) to clear or undo your drawing.")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Prediction</div>", unsafe_allow_html=True)

    if predict_clicked and canvas_result.image_data is not None:
        img = canvas_result.image_data.astype("uint8")
        img = Image.fromarray(img).convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array, verbose=0)[0]
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        st.markdown(f"<p class='result-number'>{predicted_digit}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='confidence-text'>Model confidence: <b>{confidence:.1f}%</b></p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Probability Breakdown</div>", unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Digit": [str(i) for i in range(10)],
            "Probability": prediction * 100
        }).set_index("Digit")

        st.bar_chart(prob_df, color="#7c3aed", height=220)

    elif predict_clicked:
        st.warning("Please draw a digit first.")
    else:
        st.markdown("""
            <div class="placeholder-box">
                Draw a digit on the left, then click<br><b>Predict Digit</b> to see the result here.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div class="footer-note">
        Built with a Convolutional Neural Network (CNN) trained on the MNIST dataset · 99.25% test accuracy
    </div>
""", unsafe_allow_html=True)
