# cnn-classifier
This file contains resources needed for the cnn-classifier method

# Dataset
The dataset currently being used is retrieved from keras named CIFAR-10, which is a large image dataset containing over 60,000 images representing 10 different classes of objects like cats, planes, and cars.

# Packages required
1. numpy
2. keras (use `pip install keras` to install)
3. tensorflow (use `pip install tensorflow` to install)

# Function of each file 
- `cnn-classifier.py` trains a model using 50000 images from our dataset and evaluate it using 10000 (different from training image) images. This file generates a model folder which contains information about the cnn model. Call this file when you want to generate a new model.
- `cnndescriptor.py` is used to convert an image to a vector. It takes an index as parameter then retrieve the image from our dataset using that index.
- `cnn_search.py` finds the top 10 most similar image in index.csv.
- `updatecsv.py` takes images and put them through our model to generates a csv file.

# To run each file
run `python filename`
e.g. `python cnn-classifier.py`

# Reference
https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/


