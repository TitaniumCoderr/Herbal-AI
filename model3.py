# -*- coding: utf-8 -*-
"""effnetDL.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16cOtJRSS1xNn_aDeabcEPsXIDdty4VkN
"""

import os
import shutil
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras import callbacks, optimizers
import numpy as np
from google.colab import drive
import tensorflow_datasets as tfds #imported to get the plant_village dataset
import matplotlib.pyplot as plt
import random
from tensorflow.keras.layers import GlobalAvgPool2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

drive.mount('/content/drive/')

base_folder = "/content/drive/My Drive/DL_Dataset3"
train_dir = os.path.join(base_folder, "train")
test_dir = os.path.join(base_folder, "test")
validate_dir = os.path.join(base_folder, "validate")




def img_data(dir_path, target_size, batch_size, class_lst, preprocess_func=None):
    gen_object = ImageDataGenerator(preprocessing_function=preprocess_func) if preprocess_func else ImageDataGenerator()
    return gen_object.flow_from_directory(
        dir_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        classes=class_lst,
        shuffle=True
    )

class_list = os.listdir(train_dir)  # Get class names from the train directory

train_data_gen = img_data(os.path.join(base_folder, "train"), (224, 224), 500, class_list, tf.keras.applications.efficientnet.preprocess_input)
test_data_gen = img_data(os.path.join(base_folder, "test"), (224, 224), 500, class_list, tf.keras.applications.efficientnet.preprocess_input)
valid_data_gen = img_data(os.path.join(base_folder, "validate"), (224, 224), 500, class_list, tf.keras.applications.efficientnet.preprocess_input)

# Load and set up EfficientNetB0 model
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling=None
)
base_model.trainable = False  # Freeze the base model

# Build and compile the model
model = tf.keras.Sequential([
    base_model,
    GlobalAvgPool2D(),
    BatchNormalization(),
    Dropout(0.5),  # Add dropout to prevent overfitting
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),  # Another dropout layer
    Dense(len(class_list), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Lower learning rate for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks and train the model
early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
save_ckpt = ModelCheckpoint("EffNetModel2.keras", save_best_only=True, monitor='val_loss', mode='min')

model.fit(
    train_data_gen,
    validation_data=valid_data_gen,
    epochs=10,
    callbacks=[early_stop, save_ckpt]
)

print('model saved')