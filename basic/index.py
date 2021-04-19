# USAGE
# python index.py --dataset dataset --index index.csv

# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
import glob
import cv2
from keras.datasets import cifar10


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

# open the output index file for writing
output = open("index.csv", "w")

for image in X_train:
	features = cd.describe(image)
	features = [str(f) for f in features]
	output.write(",".join(features) + "\n")
	# output.close()
output.close()