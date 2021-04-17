from pyimagesearch.cluster import Cluster
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
# ap.add_argument("-r", "--result-path", required = True,
# 	help = "Path to the result path")
args = vars(ap.parse_args())


myCluster = Cluster(args["index"])
results = myCluster.cluster()

# print(results)
