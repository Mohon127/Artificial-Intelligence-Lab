from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# -------- Polynomial definitions -------------
def eq1(x):   # y = 5x + 10
    return 5*x + 10

def eq2(x):   # y = 3x^2 + 5x + 10
    return 3*x**2 + 5*x + 10

def eq3(x):   # y = 4x^3 + 3x^2 + 5x + 10
    return 4*x**3 + 3*x**2 + 5*x + 10

# -------- Data generation ---------------------
def get_data(eq_func, n=50000):
    x = np.linspace(-10, 10, n)
    x = x / 10.0   # scale to [-1,1]
    y = eq_func(x)
    return x, y

# -------- Data preparation -------------------
def prepare_train_val_test(eq_func, n=50000):
    # Generate data
    x, y = get_data(eq_func, n)

    # Shuffle
    indices = np.arange(n)
    np.random.shuffle(indices)
    x_norm = x[indices].reshape(-1,1)
    y = y[indices]

    # Split 70/15/15
    train_n = int(n * 0.7)
    val_n   = int(n * 0.15)

    trainX, trainY = x_norm[:train_n], y[:train_n]
    valX, valY     = x_norm[train_n:train_n+val_n], y[train_n:train_n+val_n]
    testX, testY   = x_norm[train_n+val_n:], y[train_n+val_n:]

    print(f"total_n: {n}, train_n: {train_n}, val_n: {val_n}, test_n: {len(testX)}")
    return (trainX, trainY), (valX, valY), (testX, testY), x, y

# -------- Model building ---------------------
def model_1():  # Linear
    inputs = Input((1,))
    h1 = Dense(8)(inputs)
    h2 = Dense(16, activation='relu')(h1)
    outputs = Dense(1, activation='linear')(h2)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def model_2():  # Quadratic
    inputs = Input((1,))
    h1 = Dense(16, activation='relu')(inputs)
    h2 = Dense(8, activation='relu')(h1)
    outputs = Dense(1, activation='linear')(h2)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def model_3():  # Cubic
    inputs = Input((1,))
    h1 = Dense(128, activation='relu')(inputs)
    h2 = Dense(64, activation='relu')(h1)
    h3 = Dense(32, activation='relu')(h2)
    outputs = Dense(1, activation='linear')(h3)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def model_builder(eq_label):
    if eq_label == "y = 5x + 10":
        return model_1()
    elif eq_label == "y = 3x^2 + 5x + 10":
        return model_2()
    elif eq_label == "y = 4x^3 + 3x^2 + 5x + 10":
        return model_3()
    else:
        raise ValueError("Unknown equation label")

# -------- Training and plotting --------------
def train_and_plot(eq_func, eq_label, epochs=20):
    (trainX, trainY), (valX, valY), (testX, testY), x_full, y_full = prepare_train_val_test(eq_func)

    model = model_builder(eq_label)
    history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=epochs, verbose=0)

    y_pred = model.predict(testX)
    #test model 
    loss, mae = model.evaluate(testX, testY, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

    # Call plotting function
    plot_predictions(testX, testY, y_pred, eq_label, history)

# -------- Evaluation and visualization -----------
def plot_predictions(testX, testY, y_pred, eq_label, history):
    # ---- Accuracy calculation (R² score) ----
    ss_res = np.sum((testY - y_pred.flatten())**2)
    ss_tot = np.sum((testY - np.mean(testY))**2)
    r2_score = 1 - (ss_res / ss_tot)

    print(f"Accuracy (R²) for {eq_label}: {r2_score:.4f}")

    # Plot predictions vs original
    plt.scatter(testX, testY, label='Original y', color='blue', s=10)
    plt.scatter(testX, y_pred, label='Predicted y', color='red', s=10)
    plt.title(f"Equation: {eq_label} (R²={r2_score:.4f})")
    plt.xlabel("x (normalized)")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # Plot training history
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"Training Loss for {eq_label}")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

# -------- Main -------------------------------
def main():
    train_and_plot(eq1, "y = 5x + 10", epochs=5)
    train_and_plot(eq2, "y = 3x^2 + 5x + 10", epochs=20)
    train_and_plot(eq3, "y = 4x^3 + 3x^2 + 5x + 10", epochs=50)

if __name__ == "__main__":
    main()
