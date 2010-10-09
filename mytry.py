#!/usr/bin/python


import sys
from math import sin, cos, sqrt, pi
import cv
import numpy as np

sys.path.append("..")
import convert

def sharpen(im):
    # convert image depth, for Laplacian function
    im32 = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_32F, 1)
    cv.Convert(im, im32)

    # sharpen by adding the Laplacian to the image    
    lap = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_32F, 1)
    cv.Laplace(im32, lap)
    res = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_8U, 1)
    cv.Convert(lap, res)
    lap = res
    
    cv.Add(im, lap, res)
    
    return res
    
def binarizeImage(im):
    # since different parts of the image may be illuminated differently,
    # run the threshold algorithm on each part separately
    # TODO: check AdaptiveThreshold function
    imSize = cv.GetSize(im)
    bin = cv.CreateImage(imSize, 8, 1)
    numDivs = 4
    divHeight = imSize[1]//numDivs
    divWidth = imSize[0]//numDivs
    for j in range(numDivs):
        for i in range(numDivs):
            rect = (i*divWidth, j*divHeight, divWidth, divHeight)
            print rect, imSize
            cv.SetImageROI(im, rect)
            dest = cv.GetSubRect(bin, rect)
            cv.Threshold(im, dest, 0, 255, cv.CV_THRESH_BINARY_INV | cv.CV_THRESH_OTSU)
            cv.ResetImageROI(im)
    return bin

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s filename\n" % sys.argv[0])
        sys.exit(1)

    # load image
    filename = sys.argv[1]
    src = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
    cv.NamedWindow("B&W", 0)
    cv.ShowImage("B&W", src)
    
    # convert to binary
    bin = binarizeImage(src)
    cv.NamedWindow("Binary", 0)
    cv.ShowImage("Binary", bin)

    cv.WaitKey(0)
    
if __name__ == "__main__":
    main()