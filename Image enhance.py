import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Load and preprocess data
def load_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        img = load_img(os.path.join(image_folder, filename), target_size=(256, 256))
        img = img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)

underwater_images = load_images('data/underwater')
ground_truth_images = load_images('data/ground_truth')

# Build the autoencoder model
input_img = Input(shape=(256, 256, 3))

# Encoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(underwater_images, ground_truth_images, epochs=100, batch_size=16, shuffle=True)

# Enhance an underwater image
def enhance_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    enhanced_img = autoencoder.predict(img_array)
    return enhanced_img[0]

# Display the results
def display_images(original, enhanced):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original)
    axs[0].set_title('Original Underwater Image')
    axs[0].axis('off')
    axs[1].imshow(enhanced)
    axs[1].set_title('Enhanced Image')
    axs[1].axis('off')
    plt.show()

# Test the model with a sample image
sample_image_path = 'data/underwater/sample.jpg'
original_img = load_img(sample_image_path, target_size=(256, 256))
enhanced_img = enhance_image(sample_image_path)

display_images(original_img, enhanced_img)
