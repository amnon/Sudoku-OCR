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

# find intersection of two lines, given as a (rho, theta) pair
def findIntersection(line1, line2):
    # the normal to each line is (cos(theta),sin(theta)).
    # a point is on the line only if its projection on the normal
    # equals rho. hence we get two linear equations.
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = cv.CreateMat(2, 2, cv.CV_64F)
    A[0,0] = cos(theta1)
    A[1,0] = cos(theta2)
    A[0,1] = sin(theta1)
    A[1,1] = sin(theta2)
    B = cv.CreateMat(2, 1, cv.CV_64F)
    B[0,0] = rho1
    B[1,0] = rho2
    X = cv.CreateMat(2, 1, cv.CV_64F)
    cv.Solve(A, B, X)
    return X[0,0], X[1,0]

# reproject a quadrilateral region in the image to quadrangle with the
# given size, using perspective projection
def reprojectQuad(im, topLeft, bottomLeft, bottomRight, topRight, newSize):
    width, height = newSize

    # compute transform
    map = cv.CreateMat(3, 3, cv.CV_64F)
    src, dst = zip(
        (topLeft, (0,0)),
        (bottomLeft, (0,height-1)),
        (bottomRight, (width-1,height-1)),
        (topRight, (width-1,0)))
    cv.GetPerspectiveTransform(src, dst, map)

    # apply transform
    newIm = cv.CreateImage((height,width), im.depth, im.nChannels)
    cv.WarpPerspective(im, newIm, map)
    return newIm
    
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
    del imSize
    cv.Resize(im, tmp)
    im = tmp
    origImage = im

    # convert to binary
    im = binarizeImage(im)
    showImage("Binary", im)

    # get largest connected component (blob). hopefully it's the puzzle's border.
    # TODO: check if this step is really needed for the Hough transform later.
    maxBlob = getLargestBlob(im)
    showImage("Max Component", maxBlob)

    # do the Hough to get puzzle's border as lines so we can compute its corners.
    # TODO: we get too many lines because the blob is too thick. apply skeletonization/thinning?
    # the problem is that we'd like to use the lines with the highest number
    # of votes, instead of using any random line above an arbitrary threshold,
    # but OpenCV doesn't give us any indication about this.
    # maybe it's better to implement the Hough transform myself.
    rhoStep, thetaStep = 1, pi/180
    lines = cv.HoughLines2(maxBlob, cv.CreateMemStorage(), cv.CV_HOUGH_STANDARD,
        rhoStep, thetaStep, 250)
        
    # see which line(s) corresponds to which part of the border: top/left/bottom/right.
    # each line is represented as a (rho, theta) pair, where 0 <= theta < PI is
    # the angle of the normal to the line, and rho is the line's distance
    # from the origin. since the origin is at the top-left corner of the image, we
    # can use rho to know which line it is.
    width, height = cv.GetSize(im)
    for (rho, theta) in lines:
        if pi/4 < theta < pi*3/4:
            # horizontal
            if abs(rho) > height/2:
                bottomLine = (rho, theta)
            else:
                topLine = (rho, theta)
        else:
            # vertical
            if abs(rho) > width/2:
                rightLine = (rho, theta)
            else:
                leftLine = (rho, theta)

    # find intersections (corners)
    topLeft, topRight = [findIntersection(topLine, line) for line in (leftLine, rightLine)]
    bottomLeft, bottomRight = [findIntersection(bottomLine, line) for line in (leftLine, rightLine)]
    
    #  show Hough result
    color_dst = cv.CreateImage(cv.GetSize(origImage), 8, 3)
    cv.CvtColor(origImage, color_dst, cv.CV_GRAY2BGR)
    for (rho, theta) in [topLine, leftLine, bottomLine, rightLine]:
        a = cos(theta)
        b = sin(theta)
        x0 = a * rho 
        y0 = b * rho
        pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
        pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
        cv.Line(color_dst, pt1, pt2, cv.RGB(255, 0, 0), 1, 8)
    for x,y in [topLeft, topRight, bottomLeft, bottomRight]:
        cv.Circle(color_dst, (int(x), int(y)), 20, cv.RGB(255, 100, 0))
    showImage("Hough", color_dst)

    # reproject puzzle to a square image
    newWidth = newHeight = max(cv.GetSize(im))
    im = reprojectQuad(im, topLeft, bottomLeft, bottomRight, topRight, (newWidth, newHeight))
    showImage("Warped", im)
    
    cv.WaitKey(0)
    
if __name__ == "__main__":
    main()