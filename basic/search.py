# USAGE
# python search.py --index index.csv --query queries/103100.png --result-path dataset

# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher import Searcher
import argparse
import cv2
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

cd = ColorDescriptor((8, 12, 3))

query = X_train[0]
features = cd.describe(query)

# perform the search
searcher = Searcher("index.csv")
results = searcher.search(features)

# # display the query
# cv2.imshow("Query", query)

# # loop over the results
for (score, resultID) in results:
	print(resultID)
