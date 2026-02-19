'''
Problem Statement: Build a simple feedforward neural network to classify the MNIST dataset.
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). 
The task is to classify each image into one of the 10 digit classes.
Steps to follow:
1. Load the MNIST dataset using Keras.
2. Preprocess the data (normalize pixel values).
3. Build a simple feedforward neural network with at least one hidden layer.
4. Compile the model with an appropriate loss function and optimizer.
5. Train the model on the training data and validate it on the test data.
6. Evaluate the model's performance and visualize some predictions.

Note: This is a generic template for both mnist or fashion_mnist. 
You can choose either dataset by changing the dataset loading part, in the header of the code.

'''


from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets.mnist import load_data  # Change to fashion_mnist if needed
import matplotlib.pyplot as plt

dataset_name = 'mnist' # Change to 'fashion_mnist' if needed

def main():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    (train_X, train_Y), (test_X, test_Y) = load_data()
    #data_visualization(train_X, train_Y, test_X, test_Y)

    #--------------- Train model with validation data (way - 1)-------------------------------
    (train_X, train_Y), (val_X, val_Y) = validation_data(train_X, train_Y) 
    model, history = train_with_validation_data(model, train_X, train_Y, val_X, val_Y)
    #-------------------------------------------------------------------------------------------

    #----------------- Train model without manual validation spliting (way - 2) ----------------
    #history =  model.fit(train_X/255.0, train_Y, validation_split=0.1, epochs=10, batch_size=32)
    #-------------------------------------------------------------------------------------------

    #----------------------- Evaluate model -----------------------
    loss, accuracy = model.evaluate(test_X/255.0, test_Y)
    print(f"Test Accuracy: {accuracy * 100:.4f}% | Test Loss: {loss * 100:.4f}%")

    #----------------------- Plot training history ----------------------
    plot_history(history)

    prediction_grid(model, test_X, test_Y)
    prediction_condition_wise(model, test_X, test_Y)

    #----------------------- Save model ----------------------
    model_path = f"{dataset_name}_model_{accuracy:.3f}.h5"
    model.save(model_path)




#---------------------- Plotting and Evaluation ----------------------
def plot_history(history):
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


#---------------------- Training with Validation Data (Optional) ----------------------
def train_with_validation_data(model, train_X, train_Y, val_X, val_Y):
    epochs = 10
    history = model.fit(train_X/255.0, train_Y, validation_data=(val_X/255.0, val_Y), epochs=epochs, batch_size=32)
    return model, history


#---------------------- Prediction and Visualization ----------------------
def prediction_grid(model, test_X, test_Y):
    plt.figure(figsize=(12, 12))
    predictions = model.predict(test_X[:25])
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_X[i], cmap='gray')
        plt.title(f"True: {test_Y[i]}, Pred: {predictions[i].argmax()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Optional: Show predictions for only even labels (0, 2, 4, 6, 8)
def prediction_condition_wise(model, test_X, test_Y):
    # Create a grid for only even labels
    predictions = model.predict(test_X)
    even_indices = [i for i in range(len(test_Y)) if test_Y[i] % 2 == 0]
    plt.figure(figsize=(12, 12))
    total_print = 25
    count = 0
    for i in even_indices:
        if count >= total_print:
            break
        plt.subplot(5, 5, count + 1)
        plt.imshow(test_X[i], cmap='gray')
        plt.title(f"True: {test_Y[i]}, Pred: {predictions[i].argmax()}")
        plt.axis('off')
        count += 1
     
    plt.tight_layout()
    plt.show()



#-------------- Optional: Data Visualization ----------------
def data_visualization(train_X, train_Y, test_X, test_Y):
    print(f"Training data shape: {train_X.shape}, Training labels shape: {train_Y.shape}")
    print(f"Test data shape: {test_X.shape}, Test labels shape: {test_Y.shape}")
    pic = train_X[0]
    print(f"First training image shape: {pic.shape}")
    plt.imshow(pic, cmap='gray')
    plt.title(f"Label: {train_Y[0]}")
    plt.show()
    


#----------------------- Model ----------------------
def build_model():
    inputs = Input(shape=(28, 28, 1))
    h1 = Flatten()(inputs)
    h2 = Dense(128, activation='relu')(h1)
    h3 = Dense(64, activation='relu')(h2)
    outputs = Dense(10, activation='softmax')(h3)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary(show_trainable=True)
    return model


#----------------------- Validation Split (Optional) ----------------------
def validation_data(train_X, train_Y):
    # Split into training and validation sets
    val_split = 0.1
    val_n = int(len(train_X) * val_split)
    train_n = len(train_X) - val_n

    trainX, trainY = train_X[:train_n], train_Y[:train_n]
    valX, valY = train_X[train_n:], train_Y[train_n:]

    print(f"Training set: {trainX.shape}, Validation set: {valX.shape}")
    return (trainX, trainY), (valX, valY)


if __name__ == '__main__':
    main()