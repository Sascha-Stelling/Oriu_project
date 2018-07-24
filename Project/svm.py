#!/usr/bin/env python2

from __future__ import division

from hog import computeHOGDescriptors, transposeData
from CIFAR_data import extractImagesAndLabels
from helper_functions import *
from sklearn import svm

import numpy as np
import time

def svmClassification(X, y, desc):
    print "\nStarting SVM classification..."
    clf = svm.SVC()
    clf.fit(desc, y)
    return clf
    

if __name__ == "__main__":
    t0 = time.time()
    X_train, y_train = extractImagesAndLabels("cifar-10-batches-py/", "data_batch_1")
    X_test, y_test = extractImagesAndLabels("cifar-10-batches-py/", "test_batch")
    X_train = transposeImages(X_train)
    X_test = transposeImages(X_test)
    y_train = prepareLabels(y_train)
        
    t1 = time.time()
    desc_train = computeHOGDescriptors(X_train)
    clf = svmClassification(X_train, y_train, desc_train)
    t2 = time.time()
    
    print "Finished model fitting after", t2-t1, "seconds."
        
    desc_test = computeHOGDescriptors(X_test)
    
    y_predict = clf.predict(desc_test)
    t3 = time.time()
    
    print "\nFinished SVM classification after", t3-t2, "seconds."
    
    n = len(y_test)
    p = 0
    
    for i in range(0, n):
        if y_test[i] == y_predict[i]:
            p = p+1
            
    t4 = time.time()
    print "\nAccuracy of SVM classification:", (p/n)*100 , "% with", p , "correctly predicted labels and ", n, "total images.\nTotal running time:", t4-t0, "seconds."  
    