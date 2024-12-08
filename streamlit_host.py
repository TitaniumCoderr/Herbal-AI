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

    model = tf.keras.models.load_model("saved_model/myModel1.keras")
    ### load file
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    map_dict = {0:'Apple (scab)', 
                1:'Apple (Black_rot)', 
                2:'Corn (Cercospora leaf spot)', 
                3:'Blueberry (healthy)', 
                4:'Apple (Cedar apple rust)',
                5:'Cherry (healthy)', 
                6:'Apple (healthy)', 
                7:'Cherry (Powdery mildew)', 
                8:'Corn (Common rust)', 
                9:'Corn (healthy)', 
                10:'Corn (Northern Leaf Blight)', 
                11:'Grape (healthy)', 
                12:'Grape (Leaf blight / Isariopsis Leaf Spot)',
                13:'Grape (Esca / Black Measles)', 
                14:'Peach (Bacterial spot)', 
                15:'Orange (Haunglongbing / Citrus_greening)', 
                16:'Grape (Black rot)', 
                17:'Peach (healthy)', 
                18:'Bell Pepper (Bacterial spot)', 
                19:'Bell Pepper (healthy)', 
                20:'Potato (Early blight)', 
                21:'Potato (Late blight)', 
                22:'Potato (healthy)',
                23:'Raspberry (healthy)', 
                24:'Soybean (healthy)', 
                25:'Tomato (Bacterial spot)', 
                26:'Tomato (healthy)', 
                27:'Strawberry (healthy)', 
                28:'Tomato (Early blight)', 
                29:'Squash (Powdery_mildew)',
                30:'Tomato (Late_blight)', 
                31:'Tomato (Leaf_Mold)', 
                32:'Strawberry (Leaf_scorch)', 
                33:'Tomato (Two-spotted spider mite)', 
                34:'Tomato (Septoria leaf spot)', 
                35:'Tomato (Target Spot)', 
                36:'Tomato (Tomato mosaic virus)', 
                37:'Tomato (Tomato Yellow Leaf Curl Virus)'}


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
            st.success("Predicted Label for the image is {}".format(map_dict[prediction]))