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
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

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

# 4. Create a model
# here, a sequnencial model is used
model = Sequential()

# 5. Add layers
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same')) # A convolutional layer with 32 3x3 filters 
model.add(Activation('relu'))
model.add(Dropout(0.2)) # A dropout layer that randonly drops out the connections between the layers. This is to prevent overfitting
model.add(BatchNormalization()) # normalize input head into the next layer

model.add(Conv2D(64, (3, 3), padding='same')) # Another convolutional layer with more filters so that more features can be captured
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) # extract features
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten()) # flatten the data for ANN part
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3))) # densly connected layer with 256 neurons
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=maxnorm(3))) # densly connected layer with 128 neurons
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(class_num))
model.add(Activation('softmax'))

epochs = 25 # number of epochs we want to train for
optimizer = 'adam'

# 6. Compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())
