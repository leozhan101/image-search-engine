from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import csv

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

	# # code for finding k
	# model = KMeans()

	# visualizer = KElbowVisualizer(model, k=(1,30))

	# visualizer.fit(allImages)        # Fit the data to the visualizer
	# visualizer.show()        # Finalize and render the figure

	kmeans = KMeans(n_clusters=5, n_init = 100, random_state=0).fit(allImages)

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


# generate_clusteringInfo("../basic/index.csv")

generate_clusteringInfo("../cnn-classifier/index.csv")