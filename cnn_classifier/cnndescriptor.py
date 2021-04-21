# converts an image in the testing set to a vector

import numpy
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

        # 1. load model from saved_model.pd
        model = load_model('model') 
        vectors = model.predict(X_train[self.index].reshape(-1,32,32,3))
        

        return vectors
