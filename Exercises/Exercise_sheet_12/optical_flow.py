#!/usr/bin/env python2

from helper_functions import readImages, showImage
from functions_lib import compute_optical_flow
import numpy as np

if __name__ == "__main__":
    images, numImages = readImages("data/Crowd_PETS09", ".jpg")
    flow = mag = ang = [0] * len(images)
    for i in range(1, len(images)):
        flow[i], mag[i], ang[i] = compute_optical_flow(images[i], images[i-1])
