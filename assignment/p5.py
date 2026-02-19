'''
Problem Statement: Build a simple Convolutional Neural Network (CNN) to classify the MNIST dataset.
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9).
The task is to classify each image into one of the 10 digit classes.
Steps to follow:
1. Load the MNIST dataset using Keras.
2. Preprocess the data (normalize pixel values).
3. Build a CNN with at least one convolutional + pooling layer.
4. Compile the model with an appropriate loss function and optimizer.
5. Train the model on the training data and validate it on the test data.
6. Evaluate the model's performance and visualize some predictions.

Note: This template works for both mnist or fashion_mnist.
You can choose either dataset by changing the dataset loading part.
'''

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10  # Change to fashion_mnist if needed
import matplotlib.pyplot as plt

dataset_name = 'cifar10'  # Change to 'fashion_mnist', cifar10 if needed

def main():

    #---------------------- Load and preprocess data ----------------------
    (train_X, train_Y), (test_X, test_Y) = load_data()
    #data_visualization(train_X, train_Y, test_X, test_Y) 
    # Reshape and normalize data
    train_X = reshape_data(train_X)
    test_X = reshape_data(test_X)


    #---------------------- Build and compile model ----------------------
    model = build_model(dataset_name) 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    
    #------------ Run with manual validation spliting (way - 1)-------------------------------
    # Split validation set manually
    (train_X, train_Y), (val_X, val_Y) = validation_data(train_X, train_Y)    

    # Train with validation data
    model, history = train_with_validation_data(model, train_X, train_Y, val_X, val_Y)
    #-------------------------------------------------------------------------------------------


    #----------------- Train model without manual validation spliting (way - 2) ---------------
    # history =  model.fit(train_X, train_Y, validation_split=0.1, epochs=10, batch_size=32)   
    #------------------------------------------------------------------------------------------- 


    # Evaluate
    loss, accuracy = model.evaluate(test_X, test_Y)
    print(f"Test Accuracy: {accuracy * 100:.4f}% | Test Loss: {loss * 100:.4f}%")

    # Plot training history
    plot_history(history)

    # Predictions
    prediction_grid(model, test_X, test_Y)
    prediction_condition_wise(model, test_X, test_Y)

    # Save model
    model_path = f"{dataset_name}_cnn_model_{accuracy:.3f}.h5"
    model.save(model_path)


#---------------------- Data Loading ----------------------
def load_data():
    if dataset_name == 'mnist':
        return mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        return fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        return cifar10.load_data()
    else:
        raise ValueError("Unsupported dataset. Choose from 'mnist', 'fashion_mnist', or 'cifar10'.")


#---------------------- Data Preprocessing ----------------------
# Add channel dimension for compatibility with Conv2D and normalize pixel values
def reshape_data(X):
    if dataset_name in ['mnist', 'fashion_mnist']:
        return np.expand_dims(X, axis=-1) / 255.0
    elif dataset_name == 'cifar10':
        return X / 255.0
    else:
        raise ValueError("Unsupported dataset for reshaping.")


# ---------------------- Plotting ----------------------
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------- Training ----------------------
def train_with_validation_data(model, train_X, train_Y, val_X, val_Y):
    epochs = 10
    history = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=epochs, batch_size=32)
    return model, history


# ---------------------- Predictions ----------------------
def prediction_grid(model, test_X, test_Y):
    plt.figure(figsize=(12, 12))
    predictions = model.predict(test_X[:25])
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        if dataset_name in ['mnist', 'fashion_mnist']:
            plt.imshow(test_X[i].reshape(28,28), cmap='gray')
        else:
            plt.imshow(test_X[i])
        plt.title(f"True: {test_Y[i]}, Pred: {predictions[i].argmax()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def prediction_condition_wise(model, test_X, test_Y):
    predictions = model.predict(test_X)
    even_indices = [i for i in range(len(test_Y)) if test_Y[i] % 2 == 0]
    plt.figure(figsize=(12, 12))
    total_print = 25
    count = 0
    for i in even_indices:
        if count >= total_print:
            break
        plt.subplot(5, 5, count + 1)
        if dataset_name in ['mnist', 'fashion_mnist']:
            plt.imshow(test_X[i].reshape(28,28), cmap='gray')
        else:
            plt.imshow(test_X[i])
        plt.title(f"True: {test_Y[i]}, Pred: {predictions[i].argmax()}")
        plt.axis('off')
        count += 1
    plt.tight_layout()
    plt.show()


# ---------------------- Model ----------------------
def build_model(dataset_name):
    if dataset_name in ['mnist', 'fashion_mnist']:
        insput_shape = (28, 28, 1)
    elif dataset_name == 'cifar10':
        insput_shape = (32, 32, 3)


    inputs = Input(shape=insput_shape)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary(show_trainable=True)
    return model


# ---------------------- Validation Split ----------------------
def validation_data(train_X, train_Y):
    val_split = 0.1
    val_n = int(len(train_X) * val_split)
    train_n = len(train_X) - val_n
    trainX, trainY = train_X[:train_n], train_Y[:train_n]
    valX, valY = train_X[train_n:], train_Y[train_n:]
    print(f"Training set: {trainX.shape}, Validation set: {valX.shape}")
    return (trainX, trainY), (valX, valY)


#----------------------- Data Visualization (optional) ----------------------
def data_visualization(train_X, train_Y, test_X, test_Y):
    print(f"Training data shape: {train_X.shape}, Training labels shape: {train_Y.shape}")
    print(f"Test data shape: {test_X.shape}, Test labels shape: {test_Y.shape}")
    pic = train_X[0]
    print(f"First training image shape: {pic.shape}")
    plt.imshow(pic.reshape(28,28), cmap='gray')
    plt.title(f"Label: {train_Y[0]}")
    plt.show()


if __name__ == '__main__':
    main()
