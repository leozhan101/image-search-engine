# ====================================================================================================
# This file is reponsible for extracting features from the training images and write them to index.csv
# ====================================================================================================
from pyimagesearch.colordescriptor import ColorDescriptor
import glob
import cv2
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

cd = ColorDescriptor((8, 12, 3))

# Write feature info of each training image to index.csv
output = open("index.csv", "w")
for image in X_test[:100]:
	features = cd.describe(image) # extract features from each image
	features = [str(f) for f in features]
	output.write(",".join(features) + "\n")
output.close()