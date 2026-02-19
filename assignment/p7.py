import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
IMG_SIZE = (64, 64)
BATCH_SIZE = 32


def main():
    model = load_model('cnn_binary_categorical_model_3.h5')
    data_dir = '/home/mohon/Pictures/dataset_1'  
    test_generator = load_data(data_dir)

    # Evaluate model
    start_time = time.time()
    loss, accuracy = model.evaluate(test_generator)
    end_time = time.time()


    print(model.summary())
    print(f"Evaluation Time: {end_time - start_time:.2f} seconds")
    print(f"Test Accuracy: {accuracy * 100:.4f}% | Test Loss: {loss * 100:.4f}%")

    plot_predictions_grid(model, test_generator, class_names=list(test_generator.class_indices.keys()))

    

def load_data(folder_path):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        folder_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    test_generator = datagen.flow_from_directory(
        folder_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )


    print(f"Number of test samples: {test_generator.n}")
    print(f"Number of test batches: {test_generator.n // BATCH_SIZE}")

    images, labels = next(test_generator)
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

   
    

    return test_generator


def plot_predictions_grid(model, data_generator, class_names, grid_size=(3, 3)):
    images, labels = next(data_generator)


    plt.figure(figsize=(10, 10))
    for i in range(grid_size[0] * grid_size[1]):
        prediction = model.predict(images[i:i+1])
        predicted_class = class_names[np.argmax(prediction[0])]
        true_class = class_names[np.argmax(labels[i])]

        plt.subplot(grid_size[0], grid_size[1], i + 1)
        plt.imshow(images[i])
        plt.title(f"Pred: {predicted_class}\nTrue: {true_class}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()