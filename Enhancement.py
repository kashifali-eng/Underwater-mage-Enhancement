
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the dataset
# Here we use ImageDataGenerator for demonstration purposes
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    'path/to/your/dataset',  # path to dataset
    target_size=(128, 128),
    batch_size=32,
    class_mode='input',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'path/to/your/dataset',  # dataset path
    target_size=(128, 128),
    batch_size=32,
    class_mode='input',
    subset='validation'
)

# Define the autoencoder architecture
input_img = Input(shape=(128, 128, 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
history = autoencoder.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Evaluate the model
loss = autoencoder.evaluate(validation_generator)
print(f'Validation Loss: {loss}')

# Visualize some results
def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Reconstructed Image']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()

for i in range(5):
    sample = validation_generator.next()[0][0]
    reconstructed = autoencoder.predict(np.expand_dims(sample, axis=0))
    display([sample, reconstructed[0]])
