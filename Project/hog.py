#!/usr/bin/env python2

from CIFAR_data import extractImagesAndLabels
import matplotlib.pyplot as plt

from skimage.feature import hog

import numpy as np

def transposeData(images):
    images_tran = []
    for img in images:
        images_tran.append(img.asnumpy().transpose(1,2,0))
        
    return images_tran


def computeHOGDescriptors(X):
    #i = 0

    descriptors = []
    
    print "\nCalculating HOG descriptors for " , len(X), "training images..."
    
    for img in X:
        descriptors.append(hog(img, block_norm="L2", multichannel=True, visualize=False))
    
    print "Finished calculating HOG descriptors!"
    return np.asarray(descriptors)

if __name__ == "__main__":
    X_train, y_train = extractImagesAndLabels("cifar-10-batches-py/", "data_batch_1")
    X_tran = transposeData(X_train)
    desc = computeHOGDescriptors(X_tran)
    
    print desc.shape
