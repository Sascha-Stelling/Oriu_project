from hog import computeHOGDescriptors, transposeData
from CIFAR_data import extractImagesAndLabels

import numpy as np

def svmClassification(X, y, desc):
    return
    

if __name__ == "__main__":
    X_train, y_train = extractImagesAndLabels("cifar-10-batches-py/", "data_batch_1")
    desc = computeHOGDescriptors(X_train)
    
