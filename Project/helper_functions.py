#!/usr/bin/env python2

from __future__ import with_statement
from skimage.io import imread
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import sys



def readImages(path, filetype):
    # Read Images from path with each (sub)directory
    # returns: list of images found, number of images
    images = []
    print "\nReading images from " , path
    
    # Go through all subdirs to find image files
    for dirpath, dirs, files in os.walk(path, topdown=True):
        for filename in files:
            if filename.endswith(filetype):
                imagePath = os.path.join(dirpath, filename)            
                image = imread(imagePath, as_gray=True)
                if image is None:
                    print "Image " , format(imagePath) , " not read properly!"
                else:
                    # Convert image to floating point
                    image = np.float32(image)/255.0
                    images.append(image)
    
    numImages = len(images)
    
    if numImages == 0:
        print "No Images found! Aborting..."
        sys.exit(0)
    
    print "Done! " , numImages , " images found."
    
    return images, numImages
    
def transposeImage(images):
    transp_imgs = []
    for img in images:
        transp_imgs.append(img.asnumpy().transpose(1,2,0))
        
    return transp_imgs

def prepareLabels(labels):
    y_train_list = []
    for y in labels:
        if y == 0:
            y_train_list.append(0)
        elif y == 1:
            y_train_list.append(1)
        elif y == 2:
            y_train_list.append(2)
        elif y == 3:
            y_train_list.append(3)
        elif y == 4:
            y_train_list.append(4)
        elif y == 5:
            y_train_list.append(5)
        elif y == 6:
            y_train_list.append(6)
        elif y == 7:
            y_train_list.append(7)
        elif y == 8:
            y_train_list.append(8)
        elif y == 9:
            y_train_list.append(9)
            
    return y_train_list
    
def showImage(img):
    plt.imshow(img, cmap="gray")
    plt.show()
