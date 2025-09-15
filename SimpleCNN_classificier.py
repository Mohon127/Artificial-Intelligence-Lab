'''
Ten Class Classifier:
Train on custom handwritten and MNIST.
Evaluate on both test sets and visualize predictions.
'''

#======================= Imports =========================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets.mnist import load_data

num_classes = 10
img_size = (28, 28, 1)

#======================= Main Flow =========================
def main():
    # Load datasets
    custom_trainX, custom_trainY, custom_testX, custom_testY = load_custom_dataset()
    mnist_trainX, mnist_trainY, mnist_testX, mnist_testY = load_mnist_dataset()

    print(f"Custom Train : {custom_trainX.shape}, {custom_trainY.shape}")
    print(f"Custom Test  : {custom_testX.shape}, {custom_testY.shape}")
    print(f"MNIST Train  : {mnist_trainX.shape}, {mnist_trainY.shape}")
    print(f"MNIST Test   : {mnist_testX.shape}, {mnist_testY.shape}")

    # Concatenate only training sets
    combined_trainX = np.concatenate([custom_trainX, mnist_trainX], axis=0)
    combined_trainY = np.concatenate([custom_trainY, mnist_trainY], axis=0)

    # Shuffle combined training set
    indices = np.arange(len(combined_trainX))
    np.random.seed(42)
    np.random.shuffle(indices)
    combined_trainX = combined_trainX[indices]
    combined_trainY = combined_trainY[indices]

    # Build and train model
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = train_model(model, combined_trainX, combined_trainY, epochs=10, batch_size=64)

    # Evaluate separately
    print("Evaluation Results: ")
    evaluate_model(model, custom_testX, custom_testY, label="Custom Test")
    evaluate_model(model, mnist_testX, mnist_testY, label="MNIST Test")

    # Plot training history
    plot_history(history, title='Combined Training (Custom + MNIST)')




#======================= Data Loaders =========================
def load_custom_dataset(path='/home/mohon/4_1/lab/ai_lab/mnist/mnist1.npz'):
    data = np.load(path)
    trainX = data['trainX'].astype('float32') / 255.0
    trainY = to_categorical(data['trainY'], 10)
    testX = data['testX'].astype('float32') / 255.0
    testY = to_categorical(data['testY'], 10)
    return trainX, trainY, testX, testY

def load_mnist_dataset():
    (trainX, trainY), (testX, testY) = load_data()
    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)
    return trainX, trainY, testX, testY

#======================= Model Builder =========================
def build_model():
    inputs = Input(img_size)
    x = Conv2D(filters = 8, kernel_size = (3,3), activation='relu')(inputs)
    x = Conv2D(filters = 16, kernel_size = (3,3), activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(num_classes, activation = 'softmax')(x)

    model = Model(inputs, outputs)
    model.summary(show_trainable=True)

    return model

#======================= Training =========================
def train_model(model, trainX, trainY, epochs=5, batch_size=64, val_split=0.1):
    return model.fit(trainX, trainY, validation_split=val_split, epochs=epochs, batch_size=batch_size)

#======================= Evaluation =========================
def evaluate_model(model, testX, testY, label=''):
    acc = model.evaluate(testX, testY, verbose=0)[1]
    print(f"{label} Accuracy: {acc:.4f}")
    return acc

#======================= Visualization =========================
def plot_history(history, title='Training History'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


#======================= Entry Point =========================
if __name__ == '__main__':
    main()
