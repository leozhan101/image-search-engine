import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from basic.pyimagesearch.colordescriptor import ColorDescriptor
from basic.pyimagesearch.searcher import Searcher
import argparse
import cv2
from keras.datasets import cifar10

def search(index):
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()

	cd = ColorDescriptor((8, 12, 3))

	query = X_train[index]
	# query = X_test[index]
	features = cd.describe(query)

	# perform the search
	searcher = Searcher("../basic/index.csv")
	results = searcher.search(features)

	print(results)
	
	results_index = []
	
	for (score, resultID) in results:
		results_index.append(resultID)
	
	return results_index

# search(6)
