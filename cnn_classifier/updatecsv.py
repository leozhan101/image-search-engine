from keras import backend
import numpy
from keras.datasets import cifar10
from tensorflow.keras.models import load_model
from keras.utils import np_utils

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
model = load_model('model')

# Save result of dropout_2 layer to csv
layer = model.get_layer('dropout_2')
keras_function = backend.function([model.input], [layer.output])
vector = keras_function(([X_test[0:100].reshape(-1,32,32,3), 1])[0])

print(vector[0])
numpy.savetxt("index.csv", vector[0], delimiter=",")