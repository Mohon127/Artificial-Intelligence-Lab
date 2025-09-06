'''
Reverse fine-tuning: Train on custom handwritten digits, then fine-tune on MNIST.
Evaluate on both test sets and visualize predictions.
'''

#======================= Necessary Imports =========================
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets.mnist import load_data
import numpy as np
import matplotlib.pyplot as plt


#======================= Model Execution Flow =========================
def main():
  #--- Load custom dataset
  custom = np.load('mnist1.npz')
  custom_trainX = custom['trainX'].astype('float32') / 255.0
  custom_trainY = to_categorical(custom['trainY'], 10)
  custom_testX = custom['testX'].astype('float32') / 255.0
  custom_testY = to_categorical(custom['testY'], 10)

  #--- Load MNIST dataset
  (mnist_trainX, mnist_trainY), (mnist_testX, mnist_testY) = load_data()
  mnist_trainX = mnist_trainX.astype('float32') / 255.0
  mnist_testX = mnist_testX.astype('float32') / 255.0
  mnist_trainY = to_categorical(mnist_trainY, 10)
  mnist_testY = to_categorical(mnist_testY, 10)

  #--- Combine datasets for final evaluation
  combined_testX = np.concatenate([mnist_testX, custom_testX], axis=0)
  combined_testY = np.concatenate([mnist_testY, custom_testY], axis=0)

  #--- Print dataset shapes
  print("Custom Train Shape :", custom_trainX.shape, custom_trainY.shape)
  print("Custom Test Shape  :", custom_testX.shape, custom_testY.shape)
  print("MNIST Train Shape  :", mnist_trainX.shape, mnist_trainY.shape)
  print("MNIST Test Shape   :", mnist_testX.shape, mnist_testY.shape)
  print("Combined Test Shape:", combined_testX.shape, combined_testY.shape)


  #--- Build and train model on custom data
  model = build_model()
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(custom_trainX, custom_trainY, validation_split=0.1, epochs=10, batch_size=32)
  

  #--- Train on MNIST
  history = model.fit(mnist_trainX, mnist_trainY, validation_split=0.1, epochs=10, batch_size=64)

  #--- Evaluate separately
  mnist_acc = model.evaluate(mnist_testX, mnist_testY, verbose=0)[1]
  custom_acc = model.evaluate(custom_testX, custom_testY, verbose=0)[1]  
  combined_acc = model.evaluate(combined_testX, combined_testY, verbose=0)[1]


  print(f"\n MNIST Test Accuracy : {mnist_acc:.4f}")
  print(f"Custom Test Accuracy : {custom_acc:.4f}")
  print(f"Combined Test Accuracy: {combined_acc:.4f}")

  #--- Plot training history (MNIST phase)
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'], label='Train Accuracy')
  plt.plot(history.history['val_accuracy'], label='Val Accuracy')
  plt.title('MNIST Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Train Loss')
  plt.plot(history.history['val_loss'], label='Val Loss')
  plt.title('MNIST  Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  plt.tight_layout()
  plt.show()



#======================= Model Construction =========================
def build_model():
  inputs = Input((28, 28))
  x = Flatten()(inputs)
  x = Dense(64, activation='relu')(x)
  x = Dense(128, activation='relu')(x)
  x = Dense(64, activation='relu')(x)
  outputs = Dense(10, activation='softmax')(x)

  model = Model(inputs, outputs)
  model.summary(show_trainable=True)

  return model


#======================= Entry Point =========================
if __name__ == '__main__':
  main()


