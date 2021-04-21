# converts an image in the testing set to a vector

import numpy
from keras import backend
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

class CNNDescriptor:
    def __init__(self, index):
        self.index = index

    def describe(self):

        # loading in the data
     
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
            
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        # load model
        model = load_model('../cnn_classifier/model')

        # Save result of dropout_2 layer to csv
        layer = model.get_layer('dropout_2')
        keras_function = backend.function([model.input], [layer.output])
        vector = keras_function(([X_test[self.index].reshape(-1,32,32,3), 1])[0])
        print(vector[0])

        return vector[0]
