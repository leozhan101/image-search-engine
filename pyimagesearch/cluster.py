# import the necessary packages
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import csv

class Cluster:
    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath

    def cluster(self, limit = 10):
        # initialize our dictionary of results
        results = []

        # open the index file for reading
        with open(self.indexPath) as f:
            reader = csv.reader(f)
            for row in reader:
                features = [float(x) for x in row[1:]]

                results.append(features)
        f.close()

        allImages = np.array(results)

        kmeans = KMeans(n_clusters=10, n_init = 1000, random_state=0).fit(allImages)
        #
        print(kmeans.labels_)
        #
        print(kmeans.cluster_centers_)

        return 0