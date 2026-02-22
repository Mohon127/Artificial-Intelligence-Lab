import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- Parameters ----------------
batch_size = 32
epochs = 20
learning_rate = 0.001
val_split = 0.1
model_path = "../model/best_cifar10_model_2.h5"
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def main():
    # Step 1: Load and preprocess CIFAR-10 data
    train_generator, val_generator, (test_X, test_y) = load_data()

    # Step 2: Construct and compile the CNN model
    model = build_model()
    model_compile(model)

    # Step 3: Define training callbacks
    cb = callbacks()

    # Step 4: Train the model
    history = train_model(model, train_generator, val_generator, cb)

    # Step 5: Evaluate model performance on the test set
    evaluate_model(model, test_X, test_y)

    # Step 6: Visualize training and validation metrics
    plot_learning_curves(history)

    # Step 7: Display sample predictions
    sample_predictions(model, test_X, test_y)




def load_data():
    # ---------------- Load & Split Data ----------------
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()

    # Normalize [0, 1]
    train_X, test_X = train_X.astype('float32') / 255.0, test_X.astype('float32') / 255.0

    # ---------------- Data Generators ----------------
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,        
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        validation_split=val_split
    )

    train_generator = train_datagen.flow(
        train_X, train_y,
        batch_size=batch_size,
        subset='training'
    )
    val_generator = train_datagen.flow(
        train_X, train_y,
        batch_size=batch_size,
        subset='validation'
    )

    return train_generator, val_generator, (test_X, test_y)


def build_model():
    # ---------------- Model Definition ----------------
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model_compile(model):
        # ---------------- Compile Model ----------------
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

def callbacks():
    # ---------------- Callbacks ----------------
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    ]
    return callbacks

def train_model(model, train_generator, val_generator, callbacks):
    # ---------------- Train Model ----------------
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    return history


def evaluate_model(model, test_X, test_y):
    # ---------------- Evaluate ----------------
    test_loss, test_acc = model.evaluate(test_X, test_y, verbose=2)
    print(f"\nTest Accuracy: {test_acc:.4f}")


def plot_learning_curves(history):
    # ---------------- Plot Learning Curves ----------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def sample_predictions(model, test_X, test_y):
    # ---------------- Predictions ----------------
    pred_probs = model.predict(test_X)
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = test_y.flatten()

    # ---------------- Plot Sample Predictions ----------------
    num_images = 16
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        idx = np.random.randint(0, len(test_X))
        plt.subplot(4, 4, i + 1)
        plt.imshow(test_X[idx])
        plt.axis('off')

        true_label = class_names[true_classes[idx]]
        pred_label = class_names[pred_classes[idx]]
        confidence = np.max(pred_probs[idx]) * 100

        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"T:{true_label}\nP:{pred_label}\n{confidence:.1f}%", color=color, fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
