# converts an image in the testing set to a vector

import numpy
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

        # 1. loading in the data
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        print("Data loaded, X_test will be used as input.")

        # 2. normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
        X_test = X_test.astype('float32')
        X_test = X_test / 255.0

        # 3. Create a model
        # here, a sequnencial model is used
        model = Sequential()

        # 4. Add layers
        print(X_test.shape[1:])
        model.add(Conv2D(32, (3, 3), input_shape=X_test.shape[1:], padding='same')) # A convolutional layer with 32 3x3 filters 
        model.add(Activation('relu'))
        model.add(Dropout(0.2)) # A dropout layer that randonly drops out the connections between the layers. This is to prevent overfitting

        model.add(MaxPooling2D(pool_size=(3, 3))) # extract features
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2))) # extract features
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Flatten()) # flatten the data for ANN part
        model.add(Dropout(0.2))

        # 5. Run the model
        vectors = model.predict(X_train[self.index].reshape(-1,32,32,3))

        return vectors
