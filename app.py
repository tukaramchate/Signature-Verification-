import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from inference_utils import load_best_threshold
from preprocess import preprocess_signature

THRESHOLD = load_best_threshold(default=0.75)
model = load_model("best_model.h5")


def preprocess(img):
    gray = np.array(img.convert("L"))
    processed = preprocess_signature(gray)
    return np.expand_dims(processed, axis=0)


st.set_page_config(page_title="Signature Verifier", page_icon="✍️")
st.title("✍️ Signature Verification System")
st.markdown("Upload a signature to check if it is Genuine or Forged.")

uploaded = st.file_uploader("Upload Signature Image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Signature", width=300)

    if st.button("Verify Signature"):
        processed = preprocess(img)
        score = model.predict(processed, verbose=0)[0][0]
        confidence = float(score if score >= THRESHOLD else (1 - score))

        if score >= THRESHOLD:
            st.success(f"GENUINE  -  Confidence: {confidence * 100:.1f}%")
        else:
            st.error(f"FORGED  -  Confidence: {confidence * 100:.1f}%")

        st.caption(f"Decision threshold: {THRESHOLD:.2f} | Raw genuine score: {float(score):.3f}")
        st.progress(confidence)
