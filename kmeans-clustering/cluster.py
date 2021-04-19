from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import csv

results = []

with open("../basic/index.csv") as f:
	reader = csv.reader(f)
	for row in reader:
		features = [float(x) for x in row[:]]

		# print(len(features))

		results.append(features)
f.close()

allImages = np.array(results)

model = KMeans()

visualizer = KElbowVisualizer(model, k=(10,20))

visualizer.fit(allImages)        # Fit the data to the visualizer
k = visualizer.show()        # Finalize and render the figure
print(k)


# kmeans = KMeans(n_clusters=10, n_init = 100, random_state=0).fit(allImages)

# labels = kmeans.labels_

# centres = kmeans.cluster_centers_


# output = open("basic_labels.csv", "w")
# labelsInfo = [str(l) for l in labels]
# output.write(",".join(labelsInfo))
# output.close()
# #
# output = open("basic_centres.csv", "w")
# for i in range(0, len(centres)):
# 	centre = []
# 	for j in range(0, len(centres[i])):
# 		centre.append(str(centres[i][j]))
# 	output.write(",".join(centre) + "\n")
# output.close()

# output = open("basic_images.csv", "w")
# imagesInfo = [str(i) for i in images]
# output.write(",".join(imagesInfo))
# output.close()