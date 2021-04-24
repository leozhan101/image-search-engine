# ====================================================================================================
# This file is reponsible for finding the indexes of best matching images by comparing the input image 
# with every image stored in index.csv
# ====================================================================================================
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from basic.pyimagesearch.colordescriptor import ColorDescriptor
from basic.pyimagesearch.searcher import Searcher
from keras.datasets import cifar10

def search(index):
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()

	cd = ColorDescriptor((8, 12, 3))

	# Get input image and extract it's features
	query = X_test[index]
	features = cd.describe(query)

	# Pass the input into the search function
	searcher = Searcher("../basic/index.csv")
	results = searcher.search(features)
	
	results_index = []
	
	for (score, resultID) in results:
		results_index.append(resultID) # resultID is the index of the best matching image
	
	return results_index
