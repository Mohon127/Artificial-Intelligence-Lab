'''
Problem Statement:
Train a CNN to classify MNIST odd digits (1,3,5,7,9) with restricted training schedule:
- 30 epochs total
- First 10 epochs: train full network
- Next 20 epochs: freeze first 3 conv layers
- Batch size = 32, optimizer = Adagrad(lr=0.003)
- Train/val split: 85% train, 15% validation (manual split, no sklearn)
- Save best model after every epoch if val_loss improves
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# -----------------------------
# Prepare Odd Digit Dataset
# -----------------------------
def load_odd_digits():
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    # Filter odd digits
    odd_mask_train = np.isin(train_Y, [1,3,5,7,9])
    odd_mask_test = np.isin(test_Y, [1,3,5,7,9])

    train_X, train_Y = train_X[odd_mask_train], train_Y[odd_mask_train]
    test_X, test_Y = test_X[odd_mask_test], test_Y[odd_mask_test]

    # Normalize and reshape
    train_X = train_X.astype("float32")/255.0
    test_X = test_X.astype("float32")/255.0
    train_X = np.expand_dims(train_X, -1)
    test_X = np.expand_dims(test_X, -1)

    # Map labels to 0–4 (five classes)
    label_map = {1:0, 3:1, 5:2, 7:3, 9:4}
    train_Y = np.array([label_map[y] for y in train_Y])
    test_Y = np.array([label_map[y] for y in test_Y])

    return (train_X, train_Y), (test_X, test_Y)

# -----------------------------
# Manual Train/Validation Split
# -----------------------------
def split_train_val(X, y, val_ratio=0.15):
    n = len(X)
    val_n = int(n * val_ratio)
    # Shuffle indices
    indices = np.arange(n)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    X_val, y_val = X[:val_n], y[:val_n]
    X_train, y_train = X[val_n:], y[val_n:]
    return (X_train, y_train), (X_val, y_val)

# -----------------------------
# Build CNN Model
# -----------------------------
def build_model():
    inputs = Input(shape=(28,28,1))
    x = Conv2D(32, (3,3), activation='relu', name="conv1")(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', name="conv2")(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu', name="conv3")(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# -----------------------------
# Training Workflow
# -----------------------------
def main():
    (train_X, train_Y), (test_X, test_Y) = load_odd_digits()
    (X_train, y_train), (X_val, y_val) = split_train_val(train_X, train_Y, val_ratio=0.15)

    model = build_model()
    model.summary(show_trainable=True)

    for layer in model.layers[-9:-6]:
        layer.trainable = False
    
    model.summary(show_trainable=True)

    opt = tf.keras.optimizers.Adagrad(learning_rate=0.003)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Callback to save best model
    checkpoint = ModelCheckpoint("best_mnist_odd.h5", save_best_only=True, monitor="val_loss")

    # ---- First 10 epochs: train full network ----
    history1 = model.fit(X_train, y_train,
                         validation_split = 0.15,
                         epochs=10, batch_size=32,
                         callbacks=[checkpoint])

    # ---- Freeze first 3 conv layers ----
    for layer_name in ["conv1","conv2","conv3"]:
        model.get_layer(layer_name).trainable = False

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # ---- Next 20 epochs ----
    history2 = model.fit(X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=20, batch_size=32,
                         callbacks=[checkpoint])

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_X, test_Y)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
