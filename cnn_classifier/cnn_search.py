# ====================================================================================================
# This file is reponsible for finding the indexes of best matching images by comparing the input image 
# with every image stored in index.csv
# ====================================================================================================

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import csv
import argparse
import cv2
from keras.datasets import cifar10
from cnn_classifier.cnndescriptor import CNNDescriptor
from sklearn.metrics.pairwise import cosine_similarity


def search(index):
    descriptor = CNNDescriptor(index)

    queryFeatures = descriptor.describe()

    results = {}

    with open("../cnn_classifier/index.csv") as f:
        reader = csv.reader(f)

        # compare input with each image and store the similarity socre into the results
        # counter represents the index each image stored in index.csv
        counter = 0
        for row in reader:
            features = [[float(x) for x in row[:]]]
            d = cosine_similarity(features, queryFeatures)[0][0] # calculate the similarity score between input and each image
            results[counter] = d
            counter += 1

        f.close()

    # Sort the results by similarity socre in descending order
    results = sorted([(v, k) for (k, v) in results.items()], reverse=True)[:10]

    results_index = []

    for (score, resultID) in results:
        results_index.append(resultID) # resultID is the index of the best matching image
    
    return results_index
