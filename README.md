# Image Search Engine Backend

## Introduction
The main goal of this project is to explore the advantages and disadvantages of different image search methods. This project focuses on CBIR (Content-based Image Retrieval) technology, in which the computers will search for images according to the content of images instead of metadata.

## Requirments
[Python](https://www.python.org/)

## Dataset
The dataset currently being used is retrieved from keras named CIFAR-10, which is a large image dataset containing over 60,000 images representing 10 different classes of objects like cats, planes, and cars.

## Deployment
    
Before running the backend, activate vm:
```
`venv\Scripts\activate`
```

### Run backend API
```
    $ pip install -r requirements.txt
    $ cd api
    $ flask run
```

## Program structures
Programs are divided into 4 different folders:

### api
This folder contains codes that run the APIs. For instructions, please refer to the previous section. 
-  `__init__.py`: This file contains codes that calls all search methods implemented in this project. 

### basic
This folder contains codes for extracting image features based on their colours, generating an `index. csv` file to store images, and providing a search function for API calls. 
- `basic_search.py`: This file provide a search function to find the top 10 most similar image in index.csv.
- `updatecsv.py`: This file takes images and put them through `colordescriptor.py` to generates a csv file.
- `colordescriptor.py`: This file locating inside `pyimagesearch` folder is used to convert an image to a vector.

### cnn_classifier
This folder conatins codes for extracting image features based on their colours and shapes using CNN, and providing a search function for APIs
- `cnn-classifier.py`: This file trains a model using 50000 images from our dataset and evaluate it using 10000 (different from training image) images. This file generates a model folder which contains information about the cnn model. Call this file when you want to generate a new model.
- `cnndescriptor.py`: This file is used to convert an image to a vector. It takes an index as parameter then retrieve the image from our dataset using that index.
- `cnn_search.py`: This file provide a search function to find the top 10 most similar image in index.csv.
- `updatecsv.py`: This file takes images and put them through our model to generates a csv file.

### kmedoids_clustering
This folder contains codes for clustering image features based on either basic or cnn methods using kmedoids, and providing two search functions for APIs
- `cluster.py`: This file contains code for clustering image features stored in `index.csv` from either basic or cnn_classifier folder (**Note:** basic and cnn_classifier folders both contain a file called `index.csv`)
  - `basic_centres` and `basic_labels` conatins centroid and label information after clustering basic's `index.csv`
  - `cnn_centres` and `cnn_labels` conatins centroids and label information after clustering cnn_classifier's `index.csv`
- `un_basic_search.py`: This file finds the closest cluster of the input image and return the top 10 most similar images in that cluster based on the clustering results of basic's `index.csv`
- `un_cnn_search.py`: This file finds the closest cluster of the input image and return the top 10 most similar images in that cluster based on the clustering results of cnn_classifier's `index.csv`

## How to update index.csv
This project uses images from [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) 

### basic
- `updatecsv.py`: currently `index.csv` contains features extract from images stored in the variable X_test (X_test is a set of images from cifar10)
```
    // Update updatecsv.py using your prefered images then
    $ py update.csv.py
```

### cnn_classifier
```
    // Update cnn-classifier to train a new model
    $ py cnn-classifier.py

    // Update updatecsv.py using your prefered images then
    $ py update.csv.py
```

### Update kmedoids_clustering
Whenever you change `index.csv`(basic or cnn or both), you need to run `cluster.py` to ensure the clustering results are based on the latest `index.csv`
```
    $ py cluster.py
```

## Reference
https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/

https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

https://scikit-learn-extra.readthedocs.io/en/latest/modules/cluster.html#k-medoids

## License
All license are reserved to Wilfrid Laurier University
