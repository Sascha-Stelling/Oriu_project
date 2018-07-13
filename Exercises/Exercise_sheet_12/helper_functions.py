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
    
def readBboxes(path):
    # Read bounding boxes from groundtruth files
    # returns: list of bounding boxes
    crops = []
    print "\nReading bounding box data from " , path
    for dirpath, dirs, files in os.walk(path, topdown=True):
        for filename in files:
            if filename.endswith(".groundtruth"):
                cropPath = os.path.join(dirpath, filename)
                with open(cropPath, 'r') as file:
                    bbox = file.read()
                if bbox is None:
                    print "Groundtruth box coordinates not found at path " , format(imagePath)
                else:
                    nrs = []
                    for z in bbox.split(" "):
                        try:
                            nbr = float(z)
                            isnbr = True
                        except ValueError:
                            isnbr = False
                            pass
                        if isnbr:
                            # Cast to int since pixel coordinates can't be float
                            nrs.append(int(nbr))
                    
                    crops.append(nrs)
                
    if len(crops) == 0:
        print "No groundtruth box coordinates found! Aborting..."
        sys.exit(0)
                
    print "Done! ", len(crops), " bounding box data files found."
        
    return crops
    
def cropImage(img, bbox):
    # Crops image according to bounding boxes
    # Bounding box coordinates must be in order [x_min, y_min, x_max, y_max]
    # and mustn't be greater than image size
    # returns: cropped image
    
    crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    return crop_img
    
def fillCrops(images, bboxes):
    # Crop all images according to bounding boxes. 
    # Also consider multiple bounding boxes per image
    # returns: list of cropped images
    
    crops = []
    print "\nCropping images..."
    
    for i in range(len(images)):
        if len(bboxes[i]) == 4:
            crop_img = cropImage(images[i], bboxes[i])
            #print i, crop_img.shape, bboxes[i], images[i].shape
            crops.append(crop_img)
        elif len(bboxes[i]) > 4 and len(bboxes[i]) % 4 == 0:
            newbbox = [bboxes[i][x:x+4] for x in range(0, len(bboxes[i]), 4)]
            for j in range(0, len(newbbox)):
                crop_img = cropImage(images[i], newbbox[j])
                #print i, crop_img.shape, newbbox[j], images[i].shape
                crops.append(crop_img)
        else:
            print "Error: unmatched bounding box: ", bboxes[i]
        
    print crops[34]
    print "Finished cropping images. ", len(crops), " cropped Images created."
    return crops
    
def cropPatches(img, step_size):
    # Crop an image into 8x8 grid 
    # Patches are padded to square shape
    # returns: list of cropped patches (64 elements)
    maxshape = max(img.shape)
    diff_x = maxshape-img.shape[1]
    diff_y = maxshape-img.shape[0]
        
    if maxshape == img.shape[0]:
        if diff_x % 2 == 0:
            img = np.pad(img, ((0,0), (diff_x/2, diff_x/2)), mode='edge')
        else:
            img = np.pad(img, ((0,0), ((diff_x/2+1), diff_x/2)), mode='edge')
            
    else:
        if diff_y % 2 == 0:
            img = np.pad(img, (((diff_y/2), diff_y/2),(0,0)), mode='edge')
        else:
            img = np.pad(img, (((diff_y/2)+1, diff_y/2),(0,0)), mode='edge')
            
    assert img.shape[0] == img.shape[1]
        
    sample_points = []
    #print "\nCropping patches with step size ", step_size
    for c1 in range(step_size, maxshape, maxshape//step_size):
        for c2 in range(step_size, maxshape, maxshape//step_size):
            sample_points.append([c1, c2])
            
    patches = []
    for point in sample_points:
        patches.append(cropImage(img, [point[0]-step_size, point[1]-step_size, point[0]+step_size, point[1]+step_size]))
        
    #print "Finished cropping patches."
    return patches
    
def showImage(img):
    plt.imshow(img, cmap="gray")
    plt.show()