import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.optimizers import Adam

from keras.utils import load_img
from keras.preprocessing import image
from keras.layers import RandomRotation
from keras.layers import RandomContrast
from keras.layers import RandomZoom
from keras.layers import RandomFlip
from keras.layers import RandomTranslation



class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}
inv_class_mappings = {v: k for k, v in class_mappings.items()}
class_names = list(class_mappings.keys())



model = load_model('model.keras')

def load_and_preprocess_image(image_path, image_shape=(168, 168)):
    img = image.load_img(image_path, target_size=image_shape, color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array


def display_images_and_predictions(image_path, prediction, figsize=(20, 5)):
    plt.figure(figsize=figsize)
    img_array = load_and_preprocess_image(image_path)
    img_array = np.squeeze(img_array)
    plt.imshow(img_array, cmap='gray')
    title_color = 'green'
    plt.title(f'Pred: {prediction}', color=title_color)
    plt.axis('off')
    plt.show()


def Classification_Resnet(image_path):
    images = load_and_preprocess_image(image_path)
    prediction = model.predict(images)
    predicted_labels = inv_class_mappings[np.argmax(prediction)]
    return predicted_labels