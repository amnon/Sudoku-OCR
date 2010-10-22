import mytry
import convert
import cv
import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s trainingFileSet\n" % sys.argv[0])
        sys.exit(1)
    trainingFileSet = [line.rstrip() for line in open(sys.argv[1])]
    vectorSet = []
    for fname in trainingFileSet:
        handlePuzzle(fname, vectorSet)
    
    rOutput(vectorSet)
    
# print a list of (label, vector) in R format
def rOutput(vectorSet):
    for (label, vector) in vectorSet:
        print "%s,%s" % (label, ",".join([str(x) for x in vector]))

# load digits from a labeled puzzle, and add to vectorSet
def handlePuzzle(imageFilename, vectorSet):
    print >>sys.stderr, "Loading %s" % imageFilename
    digits = mytry.processImage(imageFilename, interactive=False)
    labels = mytry.readLabels(imageFilename + ".labels")
    for ((i,j), im) in digits:
        label = labels[j][i]
        if label == ' ':
            print >>sys.stderr, "Warning, cell (%d,%d) not empty but labeled as such" % (i,j)
            continue

        vectorSet.append((label, list(convert.cvImg2np(im).flat)))

if __name__ == "__main__":
    main()
    