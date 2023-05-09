import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input


_,col2,_ = st.columns([1,3,1])

with col2:

    image = Image.open('Herbal_logo_cropped.png')

    st.image(image,width=200)
    st.title("")
    st.title("Welcome to Herbal AI")

    model = tf.keras.models.load_model("saved_model/myModel.hdf5")
    ### load file
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    map_dict = {0: 'Cherry (Powdery Mildew)',
                1: 'Corn (Common Rust)',
                2: 'Corn (Healthy)',
                3: 'Grape (Black Rot)',
                4: 'Grape (Esca aka Black Measles)',
                5: 'Grape (Isariopsis Leaf Spot aka Leaf Blight)',
                6: 'Orange (Huanglongbing aka Citrus Greening)',
                7: 'Peach (Bacterial Spot)',
                8: 'Pepper Bell (Bacterial Spot)',
                9: 'Pepper Bell (Healthy)',
                10: 'Potato (Early Blight)',
                11: 'Potato (Late Blight)',
                12:'Soybean (Healthy)',
                13:'Squash (Powdery Mildew)',
                14:'Strewberry (Leaf Scorch)',
                15:'Tomato (Bacterial Spot)',
                16:'Tomato (Early Blight)',
                17:'Apple (Cedar Apple Rust)',
                18:'Apple (Black Rot)',
                19:'Apple (Healthy)',
                20:'Apple (Scab)',
                21:'Blueberry (Healthy)',
                22:'Pomegranate (Healthy)',
                23:'Janum',
                24:'Pongamia Pinnata (Healthy)'}


    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image,(224,224))
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="RGB")

        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis,...]

        Genrate_pred = st.button("Generate Prediction")    
        if Genrate_pred:
            prediction = model.predict(img_reshape).argmax()
            #st.success(prediction)
            st.success("Predicted Label for the image is {}".format(map_dict [prediction]))