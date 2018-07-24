#!/usr/bin/env python2

from hog import computeHOGDescriptors, transposeData
from CIFAR_data import extractImagesAndLabels
from helper_functions import *
from sklearn import svm

import numpy as np

def svmClassification(X, y, desc):
    print "\nStarting SVM classification..."
    clf = svm.SVC()
    clf.fit(desc, y)
    print "Finished SVM classification!"
    return clf
    

if __name__ == "__main__":
    X_train, y_train = extractImagesAndLabels("cifar-10-batches-py/", "data_batch_1")
    X_test, y_test = extractImagesAndLabels("cifar-10-batches-py/", "test_batch")
    X_train = transposeImage(X_train)
    y_train = prepareLabels(y_train)
        
    desc = computeHOGDescriptors(X_train)
    clf = svmClassification(X_train, y_train, desc)
        
    y_predict = clf.predict(X_test)
        
