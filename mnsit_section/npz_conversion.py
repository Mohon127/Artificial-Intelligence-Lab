import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Root folder containing digit subfolders 0 to 9
root_folder = '/home/mohon/4_1/code/ai_code/mnist/digits'

# Lists to hold image data and labels
images = []
labels = []

# Accepted image extensions
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Loop through digit folders
for digit in map(str, range(10)):
    digit_folder = os.path.join(root_folder, digit)
    if not os.path.isdir(digit_folder):
        continue

    for filename in os.listdir(digit_folder):
        if not filename.lower().endswith(valid_exts):
            continue

        file_path = os.path.join(digit_folder, filename)
        if not os.path.isfile(file_path):
            continue

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping invalid image: {file_path}")
            continue

        # Resize to 28x28 if needed
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        images.append(img)
        labels.append(int(digit))

# Convert to NumPy arrays
X = np.array(images, dtype=np.uint8)
y = np.array(labels, dtype=np.uint8)

# Split into train and test sets (80% train, 20% test)
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save as MNIST-style .npz file
np.savez_compressed('mnist_custom1.npz',
                    trainX=trainX,
                    trainY=trainY,
                    testX=testX,
                    testY=testY)

print(f"✅ Dataset saved as mnist_custom.npz")
print(f"   Train: {trainX.shape[0]} samples")
print(f"   Test : {testX.shape[0]} samples")
