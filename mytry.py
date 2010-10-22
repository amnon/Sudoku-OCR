#!/usr/bin/python

import sys
from math import sin, cos, sqrt, pi
import cv
import convert

def binarizeImage(im):
    # convert to binary by setting pixels darker from some threshold to 255, and
    # zeroing the others.
    # the image may not be illuminated uniformly, so apply an adaptive
    # threshold.
    # actually, OpenCV's adaptive threshold is less than perfect because param1
    # isn't relative to the average pixel values.
    blockSize = 19
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
    # approximate the contour using a polygon. this is a sort of a smoothing operation,
    # to make the lines straighter and improve the chance that the Hough transform
    # will catch them.
    # TODO: maybe it won't be needed when we finally choose lines based on quality.
    maxContour = cv.ApproxPoly(maxContour, cv.CreateMemStorage(), cv.CV_POLY_APPROX_DP, 5, 1)
    cv.DrawContours(dst, maxContour, 255, 0, -1, 1, 8)
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

def centerImage(im):
    # find center of mass, and center image around it
    width, height = cv.GetSize(im)
    moments = cv.Moments(im, binary=True)
    area = cv.GetSpatialMoment(moments, 0, 0)
    meanX = int(cv.GetSpatialMoment(moments, 1, 0)/area)
    meanY = int(cv.GetSpatialMoment(moments, 0, 1)/area)

    centered = cv.CreateImage(cv.GetSize(im), im.depth, im.nChannels)
    cv.Zero(centered)    
    
    # the corners are (0,0,width,height). (exclusive)
    # the corresponding (back-projection) points in the original image
    # are (-deltaX,-deltaY,width-deltaX,height-deltaY)
    centerX, centerY = width//2, height//2
    deltaX, deltaY = centerX - meanX, centerY - meanY
    left = max(-deltaX, 0)
    top = max(-deltaY, 0)
    right = min(width-deltaX, width)
    bottom = min(height-deltaY, height)
    srcRect = (left, top, right-left, bottom-top)
    dstRect = (left+deltaX, top+deltaY, srcRect[2], srcRect[3])
    origROI = cv.GetImageROI(im)
    cv.SetImageROI(im, (origROI[0]+srcRect[0], origROI[1]+srcRect[1]) + srcRect[2:4])
    cv.SetImageROI(centered, dstRect)
    cv.Copy(im, centered)
    cv.ResetImageROI(centered)
    cv.SetImageROI(im, origROI)
    return centered
    
# extracts a digit blob from the given box in the image.
# modifies the image so that only the blob remains.
def extractDigit(im, box):
    cv.SetImageROI(im, box)
    contour = cv.FindContours(im,
        cv.CreateMemStorage(), cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE,
        offset=box[0:2])
    
    # minimum blob area to consider. used to ignore noise in empty boxes.
    minArea = 20
    
    maxArea, maxContour = minArea, None
    
    # find largest blob
    while contour:
        try:
            # make sure we're not looking at the border,
            # by checking the distance from the center
            res = cv.PointPolygonTest(contour, (box[0]+box[2]/2, box[1]+box[3]/2), True)
            # threshold aquired by trial and error.
            # not sure why the distance is negative
            if res < -10:
                continue
                
            # check that the blob isn't too near the edge
            bounds = cv.BoundingRect(contour)
            threshold = 10
            violations = sum(bounds[i]-box[i] < threshold for i in range(2)) + \
                sum(box[i]+box[i+2]-(bounds[i]+bounds[i+2]) < threshold for i in range(2))
            if violations >= 3:
                continue
            
            area = abs(cv.ContourArea(contour))
            if area > maxArea:
                maxArea = area
                maxContour = contour
        finally:
            # this code is in "finally" to make sure the loop variable
            # is updated even after a "continue"
            contour = contour.h_next()
    cv.Zero(im)
    cv.ResetImageROI(im)    
    if not maxContour:
        # no digit found
        return None
    cv.DrawContours(im, maxContour, 255, 0, -1, cv.CV_FILLED, 8)
    cv.SetImageROI(im, box)
    centered = centerImage(im)
    cv.ResetImageROI(im)    
    return centered

# read a text file containing digit labels.
# each line in the text file represents a sudoku line.
# digits represent themselves, space char represents an empty cell.
def readLabels(fname):
    with open(fname) as file:
        return [list("%-9s" % line.replace("\n", "")) for line in file][0:9]
    
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s filename\n" % sys.argv[0])
        sys.exit(1)
    try:
        processImage(sys.argv[1], interactive=True)
    except Exception, e:
        print "Got exception:", e
    cv.WaitKey(0)

def processImage(filename, interactive):
    # load image
    im = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
    if interactive:
        showImage("B&W", im)
    
    # smooth a little since image is noisy (at least with Nokia...)
    smoothed = cv.CloneImage(im)
    cv.Smooth(im, smoothed, cv.CV_MEDIAN, 5)
    im = smoothed
    if interactive:
        showImage("Smoothed", im)
    
    # downscale
    # (for compatibility with iPhone Soduko Grab. but maybe
    # it will help to always work on the same size?)
    # FIXME: this code is now not relevant. actually, it is,
    # for the Hough threshold to be correct.
    imSize = cv.GetSize(im)
    tmp = cv.CreateImage((imSize[0]/3, imSize[1]/3), im.depth, 1)
    del imSize
    cv.Resize(im, tmp)
    im = tmp
    origImage = im

    # convert to binary
    im = binarizeImage(im)
    if interactive:
        showImage("Binary", im)

    # get largest connected component (blob). hopefully it's the puzzle's border.
    # TODO: check if this step is really needed for the Hough transform later.
    # right now it does (otherwise the transform gets random inner lines)
    maxBlob = getLargestBlob(im)
    if interactive:
        showImage("Max Component", maxBlob)

    # do the Hough to get puzzle's border as lines so we can compute its corners.
    # TODO: we get too many lines because the blob is too thick. apply skeletonization/thinning?
    # the problem is that we'd like to use the lines with the highest number
    # of votes, instead of using any random line above an arbitrary threshold,
    # but OpenCV doesn't give us any indication about this.
    # maybe it's better to implement the Hough transform myself.
    rhoStep, thetaStep = 1, pi/180
    lines = cv.HoughLines2(maxBlob, cv.CreateMemStorage(), cv.CV_HOUGH_STANDARD,
        rhoStep, thetaStep, 125)
        
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
    if interactive:
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
    newWidth = newHeight = 396
    im = reprojectQuad(origImage, topLeft, bottomLeft, bottomRight, topRight, (newWidth, newHeight))
    if interactive:
        showImage("Warped", im)

    # binarize warped image (if we warp the binary image, the result isn't binary
    # due to interpolation. if we use NN interpolation it looks bad)
    cv.AdaptiveThreshold(im, im, 255, thresholdType=cv.CV_THRESH_BINARY_INV,
        blockSize=9)
    cv.Smooth(im, im, cv.CV_MEDIAN, 3)
    if interactive:
        showImage("Warp binarized", im)

    # divide to boxes. extract digit (if any) from box.
    xs = [newWidth/9 * i for i in range(9)] + [newWidth]
    ys = [newHeight/9 * i for i in range(9)] + [newHeight]
    digits = []
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            box = (xs[i], ys[j], xs[i+1]-xs[i], ys[j+1]-ys[j])
            digit = extractDigit(im, box)
            if digit:
                digits.append(((i,j), digit))
            
    if interactive:
        tempImage = cv.CloneImage(im)
        color = 100
        for i in range(1,9):
            cv.Line(tempImage, (xs[i], 0), (xs[i], newHeight-1), color)
            cv.Line(tempImage, (0, ys[i]), (newWidth-1, ys[i]), color)        
        showImage("Extracted", tempImage)
    return digits
    
if __name__ == "__main__":
    main()