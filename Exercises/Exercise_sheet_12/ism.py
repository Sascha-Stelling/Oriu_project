#!/usr/bin/env python2

import numpy as np
from skimage.feature import hog
from skimage.transform import resize

from helper_functions import *
    
if __name__ == "__main__":
    path = "data/ETHZ/Applelogos/train/"
    images, numImages = readImages(path, ".jpg")
    bboxes = readBboxes(path)
    crops = fillCrops(images, bboxes)
    patcheslist = []
    
    
    #showImage(crops[32])
    
    for crop in crops:
        patcheslist.append(cropPatches(crop, step_size=8))
     
    codebook = []
    for patches in patcheslist:
        for patch in patches:
                try:
                    patch_rescaled = resize(patch, (40,40), anti_aliasing=False)
                    codebook.append(hog(patch_rescaled, block_norm = 'L2'))
                except ValueError:
                    break