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

    # print(cosine_similarity(queryFeatures, queryFeatures)[0][0])

    results = {}

    with open("../cnn_classifier/index.csv") as f:
        reader = csv.reader(f)

        counter = 0
        for row in reader:
            features = [[float(x) for x in row[:]]]
            d = cosine_similarity(features, queryFeatures)[0][0]
            results[counter] = d
            counter += 1

        f.close()

    results = sorted([(v, k) for (k, v) in results.items()], reverse=True)[:10]

    print(results)

    results_index = []

    for (score, resultID) in results:
        results_index.append(resultID)
    
    return results_index

# search(1)
