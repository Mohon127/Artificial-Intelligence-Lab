import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
def load_and_preprocess_mnist(sample_size=2000):
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    train_X = np.expand_dims(train_X, -1)
    test_X = np.expand_dims(test_X, -1)

    # Convert grayscale to RGB
    train_X = np.repeat(train_X, 3, axis=-1)
    test_X = np.repeat(test_X, 3, axis=-1)

    # Resize to VGG16 input size
    train_X = tf.image.resize(train_X, (224,224)).numpy()
    test_X = tf.image.resize(test_X, (224,224)).numpy()

    # Normalize
    train_X, test_X = train_X/255.0, test_X/255.0

    return (train_X, train_Y), (test_X[:sample_size], test_Y[:sample_size])

# -----------------------------
# Feature Extraction
# -----------------------------
def build_feature_extractor():
    base_vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
    feature_model = Model(inputs=base_vgg.input, outputs=Flatten()(base_vgg.output))
    return base_vgg, feature_model

def extract_features(model, data):
    return model.predict(data)

# -----------------------------
# Transfer Learning
# -----------------------------
def build_transfer_model(base_vgg):
    # Freeze most layers, fine-tune last block
    for layer in base_vgg.layers[:-4]:
        layer.trainable = False

    model = Sequential([
        base_vgg,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")  # MNIST has 10 classes
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_transfer_model(model, train_X, train_Y, epochs=2):
    history = model.fit(train_X, train_Y, epochs=epochs, batch_size=64, validation_split=0.1)
    return history

# -----------------------------
# Dimensionality Reduction & Plotting
# -----------------------------
def plot_dim_reduction(features, labels, method="PCA", title=""):
    if method == "PCA":
        reduced = PCA(n_components=2).fit_transform(features)
    elif method == "t-SNE":
        reduced = TSNE(n_components=2, random_state=42).fit_transform(features)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="tab10", s=10)
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title(title)
    plt.show()

# -----------------------------
# Main Workflow
# -----------------------------
def main():
    # Load data
    (train_X, train_Y), (test_X, test_Y) = load_and_preprocess_mnist()

    # Pre-trained feature extractor
    base_vgg, feature_model = build_feature_extractor()
    features_before = extract_features(feature_model, test_X)

    # Transfer learning
    transfer_model = build_transfer_model(base_vgg)
    train_transfer_model(transfer_model, train_X, train_Y)

    # Extract features after transfer learning (from penultimate dense layer)
    feature_model_after = Model(inputs=transfer_model.input, outputs=transfer_model.layers[-2].output)
    features_after = extract_features(feature_model_after, test_X)

    # Plot PCA & t-SNE before transfer learning
    plot_dim_reduction(features_before, test_Y, method="PCA", title="PCA Before Transfer Learning")
    plot_dim_reduction(features_before, test_Y, method="t-SNE", title="t-SNE Before Transfer Learning")

    # Plot PCA & t-SNE after transfer learning
    plot_dim_reduction(features_after, test_Y, method="PCA", title="PCA After Transfer Learning")
    plot_dim_reduction(features_after, test_Y, method="t-SNE", title="t-SNE After Transfer Learning")

if __name__ == "__main__":
    main()
