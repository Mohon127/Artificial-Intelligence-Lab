from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

def main():
    inputs = Input((2,), name='input_layer')
    h1 = Dense(4, activation='relu', name='h1')(inputs)
    h2 = Dense(2, activation='relu', name='h2')(h1)
    outputs = Dense(1, name='output_layer')(h2)

    model = Model(inputs, outputs)
    model.summary(show_trainable=True)



if __name__ == '__main__':
    main()


