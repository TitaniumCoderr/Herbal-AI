import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

# Load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("saved_model/myModel1.keras")

model = load_model()

# Define label dictionary
map_dict = {
    0: 'Apple (scab)',
    1: 'Apple (Black_rot)',
    2: 'Corn (Cercospora leaf spot)',
    3: 'Blueberry (healthy)',
    4: 'Apple (Cedar apple rust)',
    5: 'Cherry (healthy)',
    6: 'Apple (healthy)',
    7: 'Cherry (Powdery mildew)',
    8: 'Corn (Common rust)',
    9: 'Corn (healthy)',
    10: 'Corn (Northern Leaf Blight)',
    11: 'Grape (healthy)',
    12: 'Grape (Leaf blight / Isariopsis Leaf Spot)',
    13: 'Grape (Esca / Black Measles)',
    14: 'Peach (Bacterial spot)',
    15: 'Orange (Haunglongbing / Citrus_greening)',
    16: 'Grape (Black rot)',
    17: 'Peach (healthy)',
    18: 'Bell Pepper (Bacterial spot)',
    19: 'Bell Pepper (healthy)',
    20: 'Potato (Early blight)',
    21: 'Potato (Late blight)',
    22: 'Potato (healthy)',
    23: 'Raspberry (healthy)',
    24: 'Soybean (healthy)',
    25: 'Tomato (Bacterial spot)',
    26: 'Tomato (healthy)',
    27: 'Strawberry (healthy)',
    28: 'Tomato (Early blight)',
    29: 'Squash (Powdery_mildew)',
    30: 'Tomato (Late_blight)',
    31: 'Tomato (Leaf_Mold)',
    32: 'Strawberry (Leaf_scorch)',
    33: 'Tomato (Two-spotted spider mite)',
    34: 'Tomato (Septoria leaf spot)',
    35: 'Tomato (Target Spot)',
    36: 'Tomato (Tomato mosaic virus)',
    37: 'Tomato (Tomato Yellow Leaf Curl Virus)'
}

# UI Enhancements
_, col2, _ = st.columns([.5, 6, .5])

with col2:
    logo = Image.open('Herbal_logo_cropped.png')
    st.image(logo, width=200)
    st.title("Welcome to Herbal AI 🌿")
    st.write("Upload an image of a plant leaf, and we'll diagnose its health with AI!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
history = st.container()

if uploaded_file is not None:
    # Read and preprocess the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Resize to 224x224
    resized = cv2.resize(opencv_image, (224, 224))

    # Display the resized image
    st.image(resized, caption="Resized Image (224x224)", use_container_width=False, width=224)

    # Preprocess for model
    processed_image = mobilenet_v2_preprocess_input(resized)
    img_reshape = processed_image[np.newaxis, ...]

    # Prediction
    if st.button("Generate Prediction"):
        with st.spinner("Diagnosing... Please wait."):
            preds = model.predict(img_reshape)
            top_3_indices = preds[0].argsort()[-3:][::-1]  # Top 3 predictions

            # Display progress bar
            progress = st.progress(0)
            for i in range(3):
                progress.progress((i + 1) / 3)

            # Display results
            st.subheader("Results")
            for i, idx in enumerate(top_3_indices):
                st.write(f"Rank {i + 1}: **{map_dict[idx]}** with confidence {preds[0][idx]:.2f}")

            # History Section
            with history:
                st.write("---")
                st.subheader("Prediction History")
                if "history" not in st.session_state:
                    st.session_state["history"] = []

                st.session_state["history"].append(
                    {map_dict[top_3_indices[0]]: preds[0][top_3_indices[0]]}
                )

                for entry in st.session_state["history"]:
                    st.write(entry)
