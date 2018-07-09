from hog import computeHOGDescriptors, transposeData
from CIFAR_data import extractImagesAndLabels

import numpy as np

def svmClassification(X, y, desc):
    print "Starting SVM classification..."
    clf = svm.SVC()
    clf.fit(X[0:5], y[0:5])
    print "Finished SVM classification!"
    return
    

if __name__ == "__main__":
    X_train, y_train = extractImagesAndLabels("cifar-10-batches-py/", "data_batch_1")
    desc = computeHOGDescriptors(X_train)
    
