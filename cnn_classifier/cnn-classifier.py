import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import backend
from keras.datasets import cifar10

# 1. loading in the data  
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 2. normalize the inputs
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# data preperation
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

# 3. build the model
model = Sequential() # initialize model

# a convolutional layer with 32 3x3 filters
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))

# a dropout layer that randonly drops out the connections between the layers. This is to prevent overfitting
model.add(Dropout(0.2))
model.add(BatchNormalization())

# a convolutional layer with 64 3x3 filters
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) # extract features
model.add(Dropout(0.2))
model.add(BatchNormalization())

# flatten the data for ANN part
model.add(Flatten())
model.add(Dropout(0.2))

# a densly connected layer with 256 neurons
model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# a densly connected layer with 128 neurons
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# the final densly connected layer with 10 neurons
model.add(Dense(class_num))
model.add(Activation('softmax'))

epochs = 25
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

# 4 Model evaluation & save model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
model.save("model")


