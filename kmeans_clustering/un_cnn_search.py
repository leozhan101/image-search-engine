# ====================================================================================================
# This file is reponsible for finding the indexes of best matching images by finding the correponding
# cluster of the input and comparing input with every image in that cluster
# ====================================================================================================
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from cnn_classifier.cnndescriptor import CNNDescriptor
from scipy.spatial import distance
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

    # load input image and extract its features
    descriptor = CNNDescriptor(index)
    queryFeatures = descriptor.describe()

    min_distance = distance.euclidean(queryFeatures[0], centres[0])
    cluster_label = 0

    # find which cluster that input belongs to and return the label of that cluster
    # by finding the closest centroid
    for i in range(0, len(centres)):
        if distance.euclidean(queryFeatures[0], centres[i]) < min_distance:
            min_distance = distance.euclidean(queryFeatures[0], centres[i])
            cluster_label = i

    # find indexes of all images that are in the cluster
    image_position = []
    for i in range(0, len(labels)):
        if labels[i] == cluster_label:
            image_position.append(i)

    # Compare input with each image in the cluster
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

    results_index = []
        
    for (score, resultID) in results:
        results_index.append(resultID) # resultID is the index of the best matching images
        
    return results_index





