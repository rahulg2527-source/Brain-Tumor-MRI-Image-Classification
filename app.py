import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "brain_tumor_efficientnet_final.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = (224, 224)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        classes = json.load(f)

    return classes


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(IMG_SIZE)

    # Convert to NumPy
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image, image_array


# ============================================================
# LOAD MODEL + CLASSES
# ============================================================

try:

    model = load_model()
    class_names = load_class_names()

except Exception as e:

    st.error(
        f"Error loading model: {e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Brain Tumor MRI Image Classification")

st.write(
    "Upload a brain MRI image and the trained "
    "EfficientNet model will predict the class."
)

st.info(
    "⚠️ Educational/research use only. "
    "This application is not a medical diagnostic tool "
    "and should not be used for clinical decisions."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display image
    st.subheader("Uploaded MRI")

    st.image(
        image,
        caption="Input MRI",
        use_container_width=True
    )

    st.write("")

    # Prediction button
    predict_button = st.button(
        "🔍 Predict Tumor Class",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        with st.spinner(
            "Analyzing MRI image..."
        ):

            processed_image, image_array = prepare_image(
                image
            )

            # Model prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            # Predicted class index
            predicted_index = int(
                np.argmax(predictions)
            )

            # Predicted class
            predicted_class = class_names[
                predicted_index
            ]

            # Confidence
            confidence = float(
                predictions[predicted_index]
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Class: "
            f"**{predicted_class.replace('_', ' ').title()}**"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # PROBABILITY TABLE
        # ====================================================

        st.subheader(
            "Class Probabilities"
        )

        probability_data = []

        for class_name, probability in zip(
            class_names,
            predictions
        ):

            probability_data.append(
                {
                    "Class":
                        class_name.replace(
                            "_",
                            " "
                        ).title(),

                    "Probability":
                        f"{float(probability) * 100:.2f}%"
                }
            )

        st.table(
            probability_data
        )


        # ====================================================
        # PROBABILITY BAR CHART
        # ====================================================

        st.subheader(
            "Prediction Probability"
        )

        chart_data = {}

        for class_name, probability in zip(
            class_names,
            predictions
        ):

            display_name = class_name.replace(
                "_",
                " "
            ).title()

            chart_data[display_name] = float(
                probability
            )

        st.bar_chart(
            chart_data
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EfficientNetB0 Transfer Learning + Fine-Tuning | "
    "Input Size: 224 × 224"
)

st.caption(
    "For educational/research purposes only."
)