import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# -----------------------------
# Load CIFAR-10 Dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Flatten label to 1D
y_train = y_train.flatten()
y_test = y_test.flatten()

# Classes: airplane=0, automobile=1, truck=9
classes_to_keep = [0, 1, 9]
train_mask = np.isin(y_train, classes_to_keep)
test_mask = np.isin(y_test, classes_to_keep)

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

# Map labels to {0,1,2}
label_map = {0:0, 1:1, 9:2}
y_train = np.array([label_map[int(y)] for y in y_train])
y_test = np.array([label_map[int(y)] for y in y_test])

# Resize to 64x64 for VGG16
x_train = tf.image.resize(x_train, (64,64)).numpy()
x_test = tf.image.resize(x_test, (64,64)).numpy()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# -----------------------------
# Build Model with VGG16 Backbone
# -----------------------------
base_model = VGG16(input_shape=(64,64,3), include_top=False, weights="imagenet")

x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation="softmax")(x)  # 3-class classifier

model = Model(inputs=base_model.input, outputs=outputs)

# -----------------------------
# Phase 1: Transfer Learning (Freeze Backbone)
# -----------------------------
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Transfer Learning Phase (Frozen Backbone)")
model.summary()

history1 = model.fit(x_train, y_train,
                     validation_split=0.1,
                     epochs=10,
                     batch_size=32)

# -----------------------------
# Phase 2a: Fine-Tuning Whole Backbone
# -----------------------------
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Fine-Tuning Phase (Whole Backbone)")
model.summary()

history2 = model.fit(x_train, y_train,
                     validation_split=0.1,
                     epochs=10,
                     batch_size=32)

# -----------------------------
# Phase 2b: Fine-Tuning Partial Backbone
# -----------------------------
for layer in base_model.layers[:10]:   # freeze first 10 layers
    layer.trainable = False
for layer in base_model.layers[10:]:   # unfreeze deeper layers
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Fine-Tuning Phase (Partial Backbone)")
model.summary()

history3 = model.fit(x_train, y_train,
                     validation_split=0.1,
                     epochs=10,
                     batch_size=32)

# -----------------------------
# Evaluate on Test Set
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
