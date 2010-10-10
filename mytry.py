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
    # actually, OpenCV's adaptive threshold is less than perfect because param1
    # isn't relative to the average pixel values.
    blockSize = 9
    bin = cv.CloneImage(im)
    cv.AdaptiveThreshold(im, bin, 255, thresholdType=cv.CV_THRESH_BINARY_INV,
        blockSize=blockSize)
    return bin
    
def showImage(name, im):
    cv.NamedWindow(name, 0)
    cv.ShowImage(name, im)

# return a binary image containing largest connected component.
def getLargestBlob(im):
    # FindContours modifies source image, so clone it
    dst = cv.CloneImage(im)
    contour = cv.FindContours(dst,
        cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
    maxArea, maxContour = 0, None
    while contour:
        area = abs(cv.ContourArea(contour))
        if area > maxArea:
            maxArea = area
            maxContour = contour
        contour = contour.h_next()
    cv.Zero(dst)
    cv.DrawContours(dst, maxContour, 255, 0, -1, 2, 8)
    return dst

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s filename\n" % sys.argv[0])
        sys.exit(1)

    # load image
    filename = sys.argv[1]
    im = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
    showImage("B&W", im)
    
    # smooth a little since image is noisy (at least with Nokia...)
    if True:
        smoothed = cv.CloneImage(im)
        cv.Smooth(im, smoothed, cv.CV_MEDIAN, 5)
        im = smoothed
        showImage("Smoothed", im)
    
    # downscale
    # (for compatibility with iPhone Soduko Grab. but maybe
    # it will help to always work on the same size?)
    imSize = cv.GetSize(im)
    tmp = cv.CreateImage((imSize[0]/3, imSize[1]/3), im.depth, 1)
    cv.Resize(im, tmp)
    im = tmp
    
    # convert to binary
    im = binarizeImage(im)
    showImage("Binary", im)

    # get largest connected component (blob). hopefully it's the puzzle's margins.
    # TODO: check if this step is really needed for the Hough transform later.
    maxBlob = getLargestBlob(im)
    showImage("Max Component", maxBlob)

    cv.WaitKey(0)
    
if __name__ == "__main__":
    main()