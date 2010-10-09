#!/usr/bin/python


import sys
from math import sin, cos, sqrt, pi
import cv
import numpy as np

sys.path.append("..")
import convert

def binarizeImage(im):
    # convert to binary by setting pixels darker from some threshold to 255, and
    # zeroing the others.
    # the image may not be illuminated uniformly, so apply an adaptive
    # threshold.
    divisor = 9
    blockSize = min(cv.GetSize(im)) / divisor
    blockSize += 1 - blockSize % 2 # make sure it's odd
    bin = cv.CloneImage(im)
    cv.AdaptiveThreshold(im, bin, 255, thresholdType=cv.CV_THRESH_BINARY_INV,
        blockSize=blockSize)
    return bin
    
def showImage(name, im):
    cv.NamedWindow(name, 0)
    cv.ShowImage(name, im)

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s filename\n" % sys.argv[0])
        sys.exit(1)

    # load image
    filename = sys.argv[1]
    im = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
    showImage("B&W", im)
    
    # smooth a little since image is noisy (at least from Nokia...)
    smoothed = cv.CloneImage(im)
    cv.Smooth(im, smoothed, cv.CV_MEDIAN, 5)
    im = smoothed
    showImage("Smoothed", im)
    
    # convert to binary
    im = binarizeImage(im)
    showImage("Binary", im)

    cv.WaitKey(0)
    
if __name__ == "__main__":
    main()