import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10 # here, we use the cifar10 dataset, which can be imported from keras

# Set random seed for purposes of reproducibility
seed = 21

# 1. loading in the data
# The training set of the CIFAR10 dataset contains 50000 images. 
# The shape of X_train is (50000, 32, 32, 3). Each image is 32px by 32px and each pixel contains 3 dimensions (R, G, B). 
# Each value is the brightness of the corresponding color between 0 and 255.

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("data loaded ------")
print("data format ------")
print(len(X_train))
#print(X_train)
print("image format ------")
print(len(X_train[0]))
#print(X_train[0]) #[0] -> individual image
print(len(X_train[0][0]))
#print(X_train[0][0]) #[0] -> individual image
#print("Xtrain shape")
#print(X_train.shape[1])
print("input shape ------")
print(X_train.shape[1:])

# 2. normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
# this is to improve the performance of the model
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. one hot encode outputs
# images can't be used by the network as they are
# they need to be encoded first and one-hot encoding is best used when doing binary classification.
# We are effectively doing binary classification here because an image either belongs 
# to one class or it doesn't, it can't fall somewhere in-between.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]
#print(X_train[0][0])

# 4. Create a model
# here, a sequnencial model is used
model = Sequential()

# 5. Add layers
print(X_train.shape[1:])
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same')) # A convolutional layer with 32 3x3 filters 
model.add(Activation('relu'))
model.add(Dropout(0.2)) # A dropout layer that randonly drops out the connections between the layers. This is to prevent overfitting
# model.add(BatchNormalization()) # normalize input head into the next layer

# model.add(Conv2D(64, (3, 3), padding='same')) # Another convolutional layer with more filters so that more features can be captured
# model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3, 3))) # extract features
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2))) # extract features
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten()) # flatten the data for ANN part
model.add(Dropout(0.2))

#code for ANN training
#model.add(Dense(256, kernel_constraint=maxnorm(3))) # densly connected layer with 256 neurons
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
    
#model.add(Dense(128, kernel_constraint=maxnorm(3))) # densly connected layer with 128 neurons
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

#model.add(Dense(class_num))
#model.add(Activation('softmax'))

#epochs = 25 # number of epochs we want to train for
#optimizer = 'adam'

# 6. Run the model
vectors = model.predict(X_train[0:1000].reshape(-1,32,32,3))
print(vectors)
numpy.savetxt("index.csv", vectors, delimiter=",") # save result to index.csv