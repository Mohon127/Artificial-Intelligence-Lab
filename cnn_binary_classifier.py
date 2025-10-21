import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time

# Parameters
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 5


def main():
    # Set paths
    data_dir = '/home/mohon/Pictures/dataset_1'  # Dataset root directory
    # Load datasets
    train_data, val_data = dataset_loaders(data_dir)
    # Build model
    model = build_model()

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Train model
    train_time_start = time.time()
    history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
    train_time_end = time.time()
    # Save model
    model.save('cnn_binary_categorical_model_4.h5')
    # Plot training history
    train_loss_val_loss_plot(history)

    #Evaluate model
    evaluation_time_start = time.time()
    loss, accuracy = model.evaluate(val_data)
    evaluation_time_end = time.time()
    print(f"Test Accuracy: {accuracy * 100:.4f}% | Test Loss: {loss * 100:.4f}%")
    print(f"Training Time: {train_time_end - train_time_start:.2f} seconds")
    print(f"Evaluation Time: {evaluation_time_end - evaluation_time_start:.2f} seconds")

    plot_predictions_grid(model, val_data, class_names=list(val_data.class_indices.keys()))


def train_loss_val_loss_plot(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_predictions_grid(model, val_gen, class_names, num_images=16):
    
    # Get one batch of images and labels
    images, labels = next(val_gen)
    preds = model.predict(images)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)

    plt.figure(figsize=(12, 12))
    for i in range(min(num_images, len(images))):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        pred_label = class_names[pred_classes[i]]
        true_label = class_names[true_classes[i]]
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=10)
    plt.tight_layout()
    plt.show()

    

def build_model():

    inputs = Input(shape=(64, 64, 3))
    x = Conv2D(16, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()
    return model



def dataset_loaders(data_dir):

    # Data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen


if __name__ == "__main__":
    main()