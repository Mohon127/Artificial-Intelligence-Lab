'''
Problem Statement:
Design a custom Convolutional Neural Network (CNN) for handwritten digit classification.

Specifications:
a) Two CNN hidden layers of sizes 32 and 64, each followed by ReLU activation and MaxPooling2D with kernel size (3,3) and stride (1,1).
b) Use Flatten to convert feature maps into 1D, then a Dense layer of size 64, followed by an output Dense layer of size 10 with Softmax activation.
c) Display the generated CNN with the required number of parameters.
d) Use the MNIST database for training and testing.
e) Apply data augmentation (rotation, shift) to the MNIST dataset.
f) Train two models: one on the original MNIST dataset and one on the augmented dataset.
g) Use the test MNIST dataset and augmented test dataset to predict accuracy of both CNNs.
h) Compare and plot the prediction accuracy of the two CNNs.
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# -----------------------------
# Load and preprocess MNIST
# -----------------------------
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
train_X = train_X.astype("float32")/255.0
test_X = test_X.astype("float32")/255.0
train_X = np.expand_dims(train_X, -1)
test_X = np.expand_dims(test_X, -1)

# -----------------------------
# Build CNN (Functional API)
# -----------------------------
def build_cnn():
    inputs = Input(shape=(28,28,1))
    x = Conv2D(32,(3,3),activation='relu')(inputs)
    x = MaxPooling2D((3,3), strides=(1,1))(x)
    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPooling2D((3,3), strides=(1,1))(x)
    x = Flatten()(x)
    x = Dense(64,activation='relu')(x)
    outputs = Dense(10,activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# -----------------------------
# Data Augmentation
# -----------------------------
datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

# -----------------------------
# Train and Compare
# -----------------------------
def train_and_compare():
    # Model on original MNIST
    model_orig = build_cnn()
    history_orig = model_orig.fit(train_X, train_Y, epochs=2, batch_size=64,
                                  validation_data=(test_X,test_Y), verbose=1)

    # Model on augmented MNIST
    model_aug = build_cnn()
    history_aug = model_aug.fit(datagen.flow(train_X, train_Y, batch_size=64),
                                epochs=2,
                                validation_data=(test_X,test_Y), verbose=1)

    # Evaluate both
    acc_orig = model_orig.evaluate(test_X,test_Y,verbose=0)[1]
    acc_aug = model_aug.evaluate(test_X,test_Y,verbose=0)[1]
    print(f"Original CNN Test Accuracy: {acc_orig:.4f}")
    print(f"Augmented CNN Test Accuracy: {acc_aug:.4f}")

    # Plot comparison
    plt.plot(history_orig.history['val_accuracy'], label='Original Val Acc')
    plt.plot(history_aug.history['val_accuracy'], label='Augmented Val Acc')
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_and_compare()
