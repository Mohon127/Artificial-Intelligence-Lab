import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Dataset Setup (example: cats vs dogs)
# -----------------------------
train_dir = "data/train"   # replace with your dataset path
val_dir   = "data/val"

img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

val_gen = datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

# -----------------------------
# Base VGG16 Model
# -----------------------------
base_vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# -----------------------------
# Build Custom Classifier
# -----------------------------
def build_model(base_model, trainable_layers=None):
    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Optionally unfreeze some layers
    if trainable_layers == "whole":
        for layer in base_model.layers:
            layer.trainable = True
    elif isinstance(trainable_layers, int):
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# -----------------------------
# Experiment 1: Fine-tune WHOLE VGG16
# -----------------------------
model_whole = build_model(base_vgg, trainable_layers="whole")
history_whole = model_whole.fit(train_gen, validation_data=val_gen, epochs=5)

# -----------------------------
# Experiment 2: Fine-tune PARTIAL VGG16 (last 4 layers)
# -----------------------------
base_vgg_partial = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model_partial = build_model(base_vgg_partial, trainable_layers=4)
history_partial = model_partial.fit(train_gen, validation_data=val_gen, epochs=5)

# -----------------------------
# Plot Results
# -----------------------------
import matplotlib.pyplot as plt

def plot_history(history, title):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(history_whole, "Fine-tuning Whole VGG16")
plot_history(history_partial, "Fine-tuning Partial VGG16")
