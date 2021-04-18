# import the necessary packages
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import csv

class Cluster:
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def cluster(self):
        results = []
        imageName = []

        with open(self.indexPath) as f:
            reader = csv.reader(f)
            for row in reader:
                features = [float(x) for x in row[1:]]

                imageName.append(row[0])

                results.append(features)
        f.close()

        allImages = np.array(results)
        
        # 800
        kmeans = KMeans(n_clusters=10, n_init = 800, random_state=0).fit(allImages)

        return kmeans.labels_ , kmeans.cluster_centers_, imageName

