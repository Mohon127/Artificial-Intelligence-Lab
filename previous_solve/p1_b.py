'''
Problem Statement:
Implement a simple deep neural network (DNN) to approximate the polynomial:
f(x) = 7x^4 + 4x^3 - x^4 + 6   (interpreted as 7x^4 + 4x^3 - x^4 + 6)

Specifications:
- Three hidden layers: sizes 32, 64, 128
- Training samples in range [-15, +15], normalized to [-1, +1]
- 10% test, 10% validation, rest training
- Train for suitable epochs, evaluate predictions
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# -----------------------------
# Polynomial Function
# -----------------------------
def poly_func(x):
    return 7*(x**4) + 4*(x**3) - (x) + 6   # simplified: 7x^4 + 4x^3 + 6

# -----------------------------
# Generate Dataset
# -----------------------------
def generate_data(n_samples=2000):
    X = np.linspace(-15, 15, n_samples)
    np.random.shuffle(X)
    # np.random.seed(42) 
    # X = np.random.uniform(-15, 16, n_samples)
    y = poly_func(X)

    # Normalize input range [-15,15] → [-1,1]
    X_norm = X / 15.0
    return X_norm, y

# -----------------------------
# Split Dataset
# -----------------------------
def split_data(X, y):
    n = len(X)
    test_n = int(0.1 * n)
    val_n = int(0.1 * n)

    X_train, y_train = X[:n - (test_n + val_n)], y[:n - (test_n + val_n)]
    X_val, y_val = X[n - (test_n + val_n):n - test_n], y[n - (test_n + val_n):n - test_n]
    X_test, y_test = X[n - test_n:], y[n - test_n:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

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
# Plot Predictions
# -----------------------------
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
# Main Workflow
# -----------------------------
def main():
    X, y = generate_data(10000)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    model = build_dnn()
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50, batch_size=32, verbose=0)

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

    # Plot predictions
    plot_predictions(model, X_test, y_test)

if __name__ == "__main__":
    main()
