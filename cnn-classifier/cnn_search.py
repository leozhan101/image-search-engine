import numpy as np
import csv
import argparse
import cv2
from keras.datasets import cifar10


def chi2_distance(self, histA, histB, eps = 1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])
    return d

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

cd = ColorDescriptor((8, 12, 3))

query = X_train[0]
queryFeatures = cd.describe(query)


results = {}

with open("index.csv") as f:
    reader = csv.reader(f)

    counter = 0
    for row in reader:
        features = [float(x) for x in row[:]]
        d = chi2_distance(features, queryFeatures)
        results[counter] = d
        counter += 1

    # close the reader
    f.close()

results = sorted([(v, k) for (k, v) in results.items()])[:10]

# # loop over the results
for (score, resultID) in results:
	print(resultID)
