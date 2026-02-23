'''
Problem Statement:
Train a binary classifier using transfer learning and fine-tuning with VGG16 on the CIFAR-10 dataset.

Specifications:
a) Convert CIFAR-10 labels into binary classes (e.g., even vs odd class indices).
b) Use VGG16 pretrained on ImageNet as the backbone (include_top=False).
c) Add custom classifier head: Flatten → Dense(256, ReLU) → Dense(2, Softmax).
d) Phase 1 (Transfer Learning): Freeze backbone, train only new head.
e) Phase 2 (Fine-Tuning): Unfreeze some or all convolutional layers, retrain with small learning rate.
f) Compare performance between (a) whole frozen backbone and (b) partially unfrozen backbone.
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# -----------------------------
# Load and preprocess CIFAR-10
# -----------------------------
(trainX, trainY), (testX, testY) = cifar10.load_data()

# Binary labels: even=0, odd=1
trainY = (trainY % 2).flatten()
testY = (testY % 2).flatten()

# One-hot encode for softmax output
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# Resize CIFAR-10 images to 64x64 (lighter than 224x224)
trainX = tf.image.resize(trainX, (64,64)).numpy()
testX = tf.image.resize(testX, (64,64)).numpy()

trainX = trainX.astype("float32")/255.0
testX = testX.astype("float32")/255.0

# -----------------------------
# Build VGG16-based classifier
# -----------------------------
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(64,64,3))

x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# -----------------------------
# Phase 1: Transfer Learning (freeze backbone)
# -----------------------------
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("=== Transfer Learning Phase ===")
history_tl = model.fit(trainX, trainY,
                       validation_data=(testX,testY),
                       epochs=10, batch_size=64, verbose=1)

# -----------------------------
# Phase 2: Fine-Tuning (unfreeze some layers)
# -----------------------------
for layer in base_model.layers[-4:]:  # unfreeze last 4 conv layers
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-4),  # smaller LR
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("=== Fine-Tuning Phase ===")
history_ft = model.fit(trainX, trainY,
                       validation_data=(testX,testY),
                       epochs=10, batch_size=64, verbose=1)

# -----------------------------
# Compare Results
# -----------------------------
plt.plot(history_tl.history['val_accuracy'], label='Transfer Learning Val Acc')
plt.plot(history_ft.history['val_accuracy'], label='Fine-Tuning Val Acc')
plt.title("Validation Accuracy Comparison (Odd vs Even CIFAR-10)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
