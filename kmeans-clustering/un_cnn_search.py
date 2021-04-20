import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# from basic.pyimagesearch.colordescriptor import ColorDescriptor
# from scipy.spatial.distance import cdist
import numpy as np
from keras.datasets import cifar10
import cv2
import csv

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

# def chi2_distance(histA, histB, eps=1e-10):
#     d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
#                       for (a, b) in zip(histA, histB)])

#     return d

labels = open_csv("cnn_labels.csv")
centres = open_csv("cnn_centres.csv")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = X_train[0]
queryFeatures = cd.describe(query)


min_distance = chi2_distance(queryFeatures, centres[0])
cluster_label = 0

# find which cluster query belongs to and return the label of that cluster
for i in range(1, len(centres)):
    if chi2_distance(queryFeatures, centres[i]) < min_distance:
        min_distance = chi2_distance(queryFeatures, centres[i])
        cluster_label = i


# # find indexes of all images that are in the target cluster
image_position = []
for i in range(0, len(labels)):
    if labels[i] == cluster_label:
        image_position.append(i)


results = {}
with open("../cnn-classifier/index.csv") as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter in image_position:
            features = [float(x) for x in row[:]]
            d = chi2_distance(features, queryFeatures)
            results[counter] = d
        counter += 1
    f.close()


results = sorted([(v, k) for (k, v) in results.items()])[:10]

# # # loop over the results
for (score, resultID) in results:
	print(resultID)







