# USAGE
# python index.py --dataset dataset --index index.csv

# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.cluster import Cluster
import argparse
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
# ap.add_argument("-kp", "--kPath", required = True,
# 	help = "Path to where the kmeans centres, labels, etc will be stored ")
args = vars(ap.parse_args())

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

# open the output index file for writing
output = open(args["index"], "w")

# use glob to grab the image paths and loop over them
for imagePath in glob.glob(args["dataset"] + "/*.png"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	imageID = imageID[8:]

	image = cv2.imread(imagePath)

	# describe the image
	features = cd.describe(image)
	
	# write the features to file
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))
	
# close the index file
output.close()


myCluster = Cluster(args["index"])
labels, centres, images = myCluster.cluster()

output = open("clusteringInfo/labels.csv", "w")
labelsInfo = [str(l) for l in labels]
output.write(",".join(labelsInfo))
output.close()
#
output = open("clusteringInfo/centres.csv", "w")
for i in range(0, len(centres)):
	centre = []
	for j in range(0, len(centres[i])):
		centre.append(str(centres[i][j]))
	output.write(",".join(centre) + "\n")
output.close()

output = open("clusteringInfo/images.csv", "w")
imagesInfo = [str(i) for i in images]
output.write(",".join(imagesInfo))
output.close()