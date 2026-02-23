import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# -----------------------------
# Polynomial function
# -----------------------------
def poly_func(x):
    return 5*(x**3) - 10*(x**2) - 20*x + 10

# -----------------------------
# Generate dataset
# -----------------------------
def generate_data(n_samples=2000):
    X = np.linspace(-20, 20, n_samples)
    np.random.shuffle(X)
    y = poly_func(X)

    # Normalize input range [-20,20] → [-1,1]
    X_norm = X / 20.0
    return X_norm, y

# -----------------------------
# Manual split: 90% train, 5% val, 5% test
# -----------------------------
def split_data(X, y):
    n = len(X)
    test_n = int(0.05*n)
    val_n = int(0.05*n)

    # Shuffle indices
    indices = np.arange(n)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    X_test, y_test = X[:test_n], y[:test_n]
    X_val, y_val = X[test_n:test_n+val_n], y[test_n:test_n+val_n]
    X_train, y_train = X[test_n+val_n:], y[test_n+val_n:]
    return (X_train,y_train), (X_val,y_val), (X_test,y_test)

# -----------------------------
# Build DNN (Functional API)
# -----------------------------
def build_dnn():
    inputs = Input(shape=(1,))
    h1 = Dense(32, activation='relu')(inputs)
    h2 = Dense(64, activation='relu')(h1)
    h3 = Dense(128, activation='relu')(h2)
    outputs = Dense(1)(h3)   # regression output
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    return model

# -----------------------------
# Plot training curves
# -----------------------------
def plot_history(history):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title("Training vs Validation Accuracy (MAE)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training vs Validation Error (MSE)")
    plt.legend()
    plt.show()

#-----------------------------
# Plot predictions vs true values
#-----------------------------
def plot_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8,6))
    plt.scatter(X_test, y_test, label="True", color="blue", s=10)
    plt.scatter(X_test, y_pred, label="Predicted", color="red", s=10)
    plt.title("Prediction vs True Levels (Test Data)")
    plt.xlabel("Normalized x")
    plt.ylabel("Polynomial Value")
    plt.legend()
    plt.show()

# -----------------------------
# Main workflow
# -----------------------------
def main():
    X, y = generate_data(10000)
    (X_train,y_train), (X_val,y_val), (X_test,y_test) = split_data(X,y)

    model = build_dnn()
    history = model.fit(X_train, y_train,
                        validation_data=(X_val,y_val),
                        epochs=50, batch_size=32, verbose=1)

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss={loss:.4f}, Test MAE={mae:.4f}")

    # Plot curves
    plot_history(history)
    plot_predictions(model, X_test, y_test)
if __name__ == "__main__":
    main()
