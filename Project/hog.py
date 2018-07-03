#!/usr/bin/env python2
from CIFAR_data import extractImagesAndLabels
import matplotlib.pyplot as plt

from skimage.feature import hog

import numpy as np


def computeHOGDescriptors(path, file):
    X_train, y_train = extractImagesAndLabels(path, file)

    i = 0

    descriptors = []
    
    print "Calculating HOG descriptors for " , X_train.shape[0], "training images..."
    for img in X_train:
        img = X_train[i]
        img = img.asnumpy().transpose(1,2,0)
        descriptors.append(hog(img, multichannel=True))
        i = i + 1
        
    print "Finished calculating HOG descriptors!"
    return np.asarray(descriptors)

if __name__ == "__main__":
    desc = computeHOGDescriptors("cifar-10-batches-py/", "data_batch_1")
    
    print desc.shape