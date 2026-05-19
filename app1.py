import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# LOAD MODEL
# =========================
from tensorflow import keras

model = keras.models.load_model("dog_cat_cnn_model.keras")

# =========================
# UI
# =========================
st.set_page_config(page_title="Dog vs Cat Detector", page_icon="🐶🐱")

st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Upload only DOG or CAT images. Others will be rejected.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(img):
    img = img.resize((150, 150))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess(img)

    # =========================
    # PREDICTION
    # =========================
    pred = model.predict(img_array)[0]

    # CASE 1: sigmoid output (1 neuron)
    if len(pred) == 1:
        dog_prob = float(pred[0])
        cat_prob = 1 - dog_prob
    else:
        # CASE 2: softmax output (2 neurons)
        cat_prob = float(pred[0])
        dog_prob = float(pred[1])

    confidence = max(dog_prob, cat_prob)

    # =========================
    # STRICT FILTER (IMPORTANT FIX)
    # =========================

    # 🚨 Reject low confidence / unknown images
    if confidence < 0.80:
        st.error("❌ Unknown Image! Please upload ONLY clear DOG or CAT images.")
    
    else:
        if dog_prob > cat_prob:
            st.success(f"🐶 DOG detected ({dog_prob*100:.2f}% confidence)")
        else:
            st.success(f"🐱 CAT detected ({cat_prob*100:.2f}% confidence)")