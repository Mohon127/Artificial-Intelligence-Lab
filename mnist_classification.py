'''
Mnist data, classification problem.
'''

#======================= Necessary Imports =========================
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical


#======================= Model Execution Flow =========================
def main():
  #--- Build model
  model = build_model()

  #--- Compile model with Adam optimizer and MSE loss
  model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


  #--- Load data
  (trainX, trainY),(testX, testY) = load_data()
	
  #--- Cross-check 
  print(trainX.shape, trainY.shape)
  print(testX.shape, testY.shape)
  trainX = trainX.astype('float32') / 255.0
  testX = testX.astype('float32') / 255.0
  trainY = to_categorical(trainY, num_classes=10)
  testY = to_categorical(testY, num_classes=10)

  #--- Train model
  history = model.fit(trainX, trainY, validation_split = 0.1, epochs = 10)

  #--- Predict on test data
  
  history = model.evaluate(testX, testY)
  
  plt.figure(figsize = (20,20))
  plt.subplot(2,1,1)
  plt.plot()





#======================= Model Construction =========================
def build_model():
    inputs = Input((28, 28), name='input_layer')
    x = Flatten(name='flatten')(inputs)
    x = Dense(64, activation='relu', name='dense_1')(x)
    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dense(64, activation='relu', name='dense_3')(x)
    outputs = Dense(10, activation='softmax', name='output_layer')(x)

    model = Model(inputs, outputs, name='mnist_classifier')
    model.summary()
    return model




#======================= Entry Point =========================
if __name__ == '__main__':
  main()
