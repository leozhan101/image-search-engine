# ====================================================================================================
# This file is reponsible for clustering data stored in index.csv and store centroid and label 
# information into prefix_centres.csv and prefix_labels.csv
# ====================================================================================================
from sklearn.cluster import KMeans
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import csv
import math
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
import warnings
from pylab import rcParams

def find_k(allImages, numberOfImages):
	sil_results = {}
	kmax = numberOfImages

	# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
	for k in range(2, kmax):
		kmeans = KMeans(n_clusters = k).fit(allImages)
		labels = kmeans.labels_
		sil_results[k] =  metrics.silhouette_score(allImages, labels)
	
	k = sorted([(v, k) for (k, v) in sil_results.items()], reverse=True)[0][1]
	
	return k
	

def generate_clusteringInfo(filePath):
	results = []

	numberOfImages = 0
    # store features of each image into the results
	with open(filePath) as f:
		reader = csv.reader(f)
		for row in reader:
			features = [float(x) for x in row[:]]
			results.append(features)
			numberOfImages += 1
	f.close()

	# store results as an np array
	allImages = np.array(results)
	
	k = find_k(allImages, numberOfImages)

	print("k here: ", k)

	kmeans = KMeans(n_clusters=k, random_state=0).fit(allImages)

	labels = kmeans.labels_

	centres = kmeans.cluster_centers_

	# Define prefix of the lables and centres file
	prefix = "basic"
	if "cnn" in filePath:
		prefix = "cnn"

	# write centroid and label information to prefix_lables and prefix_centres file
	output = open(prefix + "_labels.csv", "w")
	labelsInfo = [str(l) for l in labels]
	output.write(",".join(labelsInfo))
	output.close()

	output = open(prefix + "_centres.csv", "w")
	for i in range(0, len(centres)):
		centre = []
		for j in range(0, len(centres[i])):
			centre.append(str(centres[i][j]))
		output.write(",".join(centre) + "\n")
	output.close()

	SC = metrics.silhouette_score(allImages, labels)
	print(prefix, "--Silhouette Coefficient: ", SC)


# generate_clusteringInfo("../basic/index.csv")

generate_clusteringInfo("../cnn_classifier/index.csv")