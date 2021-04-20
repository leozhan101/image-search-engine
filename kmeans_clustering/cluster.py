from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import csv


import math
import matplotlib.pylab as plt
# import numpy as np
# import seaborn as sns
from sklearn.cluster import KMeans
import warnings
from pylab import rcParams


def calc_distance(x1, y1, a, b, c):
	d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
	return d

def find_k(data):
	dist_points_from_cluster_centre = []
	K = range(2, 30)
	for no_of_clusters in K:
		k_model = KMeans(n_clusters=no_of_clusters)
		k_model.fit(data)
		dist_points_from_cluster_centre.append(k_model.inertia_)

	a = dist_points_from_cluster_centre[0] - dist_points_from_cluster_centre[-1]
	b = K[-1] - K[0]
	c1 = K[0] * dist_points_from_cluster_centre[-1]
	c2 = K[-1] * dist_points_from_cluster_centre[0]
	c = c1 - c2

	distance_of_points_from_line = []
	for k in range(len(dist_points_from_cluster_centre)):
		distance_of_points_from_line.append(
			calc_distance(K[k], dist_points_from_cluster_centre[k], a, b, c))
	
	k = distance_of_points_from_line.index(max(distance_of_points_from_line)) + 1
	
	return k
	

def generate_clusteringInfo(filePath):
	results = []

	with open(filePath) as f:
		reader = csv.reader(f)
		for row in reader:
			features = [float(x) for x in row[:]]

			# print(len(features))

			results.append(features)
	f.close()

	allImages = np.array(results)

	k = find_k(allImages)

	print(k)

	# # code for finding k
	# model = KMeans()

	# visualizer = KElbowVisualizer(model, k=(1,30))

	# visualizer.fit(allImages)        # Fit the data to the visualizer
	# visualizer.show()        # Finalize and render the figure

	kmeans = KMeans(n_clusters=k, n_init = 100, random_state=0).fit(allImages)

	labels = kmeans.labels_

	centres = kmeans.cluster_centers_


	prefix = "basic"

	if "cnn" in filePath:
		prefix = "cnn"


	output = open(prefix + "_labels.csv", "w")
	labelsInfo = [str(l) for l in labels]
	output.write(",".join(labelsInfo))
	output.close()
	#
	output = open(prefix + "_centres.csv", "w")
	for i in range(0, len(centres)):
		centre = []
		for j in range(0, len(centres[i])):
			centre.append(str(centres[i][j]))
		output.write(",".join(centre) + "\n")
	output.close()


generate_clusteringInfo("../basic/index.csv")

generate_clusteringInfo("../cnn_classifier/index.csv")