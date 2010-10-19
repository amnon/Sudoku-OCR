import mytry
import cv
import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s filename\n" % sys.argv[0])
        sys.exit(1)
    handlePuzzle(sys.argv[1])
    cv.WaitKey(0)
    
    
def handlePuzzle(imageFilename):
    digits = mytry.processImage(imageFilename, interactive=False)
    labels = mytry.readLabels(imageFilename + ".labels")
    for ((i,j), im) in digits:
        label = labels[j][i]
        if label == ' ':
            print "Warning, cell (%d,%d) not empty but labeled as such" % (i,j)
            continue

        mytry.showImage(label, im)

if __name__ == "__main__":
    main()