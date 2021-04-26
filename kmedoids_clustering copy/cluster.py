# ====================================================================================================
# This file is reponsible for clustering data stored in index.csv and store centroid and label 
# information into prefix_centres.csv and prefix_labels.csv
# ====================================================================================================
from sklearn import metrics
import numpy as np
import csv
from sklearn_extra.cluster import KMedoids

def chi2_distance(histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d

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

	# Define prefix of the lables and centres file
	prefix = "basic"
	if "cnn" in filePath:
		prefix = "cnn"

	# store results as an np array
	allImages = np.array(results)
	
	if prefix == "basic":
		print("for basic")
		Kmedoids = KMedoids(n_clusters=10, metric=chi2_distance, method='pam', random_state=0).fit(allImages)
	else:
		print("for cnn")
		Kmedoids = KMedoids(n_clusters=10, metric='cosine', method='pam', random_state=0).fit(allImages)

	labels = Kmedoids.labels_

	centres = Kmedoids.cluster_centers_

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
	print(prefix, "Silhouette Coefficient: ", SC)


generate_clusteringInfo("../basic/index.csv")

# generate_clusteringInfo("../cnn_classifier/index.csv")