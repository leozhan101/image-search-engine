from pyimagesearch.colordescriptor import ColorDescriptor
from scipy.spatial.distance import cdist
import numpy as np
import argparse
import cv2
import csv

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--index", required = True,
# 	help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
args = vars(ap.parse_args())

def open_csv(filePath):
    info = []
    with open(filePath) as f:
        reader = csv.reader(f)
        for row in reader:
            if "labels" in filePath:
                info = [int(x) for x in row[:]]
            elif "centres" in filePath:
                centre = [float(x) for x in row[:]]
                info.append(centre)
            else:
                info = [str(x) for x in row[:]]

        f.close()
    return info

def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    return d

labels = open_csv("clusteringInfo/labels.csv")
images = open_csv("clusteringInfo/images.csv")
centres = open_csv("clusteringInfo/centres.csv")

# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = cv2.imread(args["query"])
queryFeatures = cd.describe(query)


min_distance = chi2_distance(queryFeatures, centres[0])
cluster_label = 0

for i in range(1, len(centres)):
    if chi2_distance(queryFeatures, centres[i]) < min_distance:
        min_distance = chi2_distance(queryFeatures, centres[i])
        cluster_label = i

image_position = []
for i in range(0, len(labels)):
    if labels[i] == cluster_label:
        image_position.append(i)


results = {}
with open("index.csv") as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter in image_position:
            features = [float(x) for x in row[1:]]
            d = chi2_distance(features, queryFeatures)
            results[row[0]] = d
        counter += 1
    f.close()

results = sorted([(v, k) for (k, v) in results.items()])[:10]


cv2.imshow("Query", query)
#
# # loop over the results
for (score, resultID) in results:
	# load the result image and display it
	result = cv2.imread(args["result_path"] + "/" + resultID)
	cv2.imshow("Result", result)
	cv2.waitKey(0)







