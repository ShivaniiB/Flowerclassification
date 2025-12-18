import os
import io
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ---------- Paths ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "flower_cnn.keras")

IMG_SIZE = (224, 224)

# ⚠️ MUST MATCH train_ds.class_names order printed during training
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ---------- Preprocess ----------
def preprocess_image(pil_img):
    # Ensure RGB
    pil_img = pil_img.convert("RGB")

    # Resize to model input
    pil_img = pil_img.resize(IMG_SIZE)

    # Convert to numpy array
    img_array = np.array(pil_img).astype("float32")

    # Add batch dimension → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # ❌ DO NOT divide by 255 here
    # Model already has Rescaling(1./255)

    return img_array

# ---------- App ----------
def main():
    st.title("🌸 Flower Classification (CNN)")
    st.write("Upload a flower image to classify it.")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes))

        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                model = load_model()
                x = preprocess_image(pil_img)

                preds = model.predict(x)[0]  # shape: (5,)
                class_idx = int(np.argmax(preds))
                confidence = float(preds[class_idx])

                st.success(
                    f"Prediction: **{CLASS_NAMES[class_idx]}** "
                    f"(confidence: {confidence:.2f})"
                )

                st.write("Class probabilities:")
                for i, cls in enumerate(CLASS_NAMES):
                    st.write(f"{cls}: {preds[i]:.3f}")

if __name__ == "__main__":
    main()
