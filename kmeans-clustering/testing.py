import math
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
from pylab import rcParams

dist_points_from_cluster_centre = []
K = range(2, 30)
for no_of_clusters in K:
    k_model = KMeans(n_clusters=no_of_clusters)
    k_model.fit(X)
    dist_points_from_cluster_centre.append(k_model.inertia_)
