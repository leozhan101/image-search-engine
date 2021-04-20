import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# from basic.pyimagesearch.colordescriptor import ColorDescriptor
from cnn_classifier.cnndescriptor import CNNDescriptor
# from scipy.spatial.distance import cdist
import numpy as np
from keras.datasets import cifar10
import cv2
import csv
from sklearn.metrics.pairwise import cosine_similarity

def open_csv(filePath):
    info = []
    with open(filePath) as f:
        reader = csv.reader(f)
        for row in reader:
            if "labels" in filePath:
                info = [int(x) for x in row[:]]
            else:
                centre = [float(x) for x in row[:]]
                info.append(centre)
        f.close()
    return info


def search(index):
    labels = open_csv("../kmeans_clustering/cnn_labels.csv")
    centres = open_csv("../kmeans_clustering/cnn_centres.csv")

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # initialize the image descriptor
    descriptor = CNNDescriptor(index)
    queryFeatures = descriptor.describe()


    min_distance = cosine_similarity(queryFeatures, [centres[0]])[0][0]
    cluster_label = 0

    # find which cluster query belongs to and return the label of that cluster
    for i in range(1, len(centres)):
        if cosine_similarity(queryFeatures, [centres[i]])[0][0] < min_distance:
            min_distance = cosine_similarity(queryFeatures, [centres[i]])[0][0]
            cluster_label = i


    # # find indexes of all images that are in the target cluster
    image_position = []
    for i in range(0, len(labels)):
        if labels[i] == cluster_label:
            image_position.append(i)


    results = {}
    with open("../cnn_classifier/index.csv") as f:
        reader = csv.reader(f)
        counter = 0
        for row in reader:
            if counter in image_position:
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







