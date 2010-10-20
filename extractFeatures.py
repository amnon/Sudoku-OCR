import mytry
import convert
import cv
import sys
import numpy as np
import bpnn

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s trainingFileSet\n" % sys.argv[0])
        sys.exit(1)
    trainingFileSet = [line.rstrip() for line in open(sys.argv[1])]
    train = []
    for fname in trainingFileSet:
        handlePuzzle(fname, train)
    
    # convert to bpnn.py format.
    # since this is a neural network, encode each digit as a binary vector
    # which is all zeros except 1 at the corresponding 
    train = [(inputs, (np.arange(1,10) == int(label))*1) for label, inputs in train]
    
    numInput = len(train[0][0])
    numHidden = 10
    numOutput = len(train[0][1])
    print numInput, numHidden, numOutput
    nn = bpnn.NN(numInput, numHidden, numOutput)
    nn.train(train)
    nn.test(train)
    
# load digits from a labeled puzzle, and add to vectorSet
def handlePuzzle(imageFilename, vectorSet):
    print "Loading %s" % imageFilename
    digits = mytry.processImage(imageFilename, interactive=False)
    labels = mytry.readLabels(imageFilename + ".labels")
    for ((i,j), im) in digits:
        label = labels[j][i]
        if label == ' ':
            print "Warning, cell (%d,%d) not empty but labeled as such" % (i,j)
            continue

        vectorSet.append((label, list(convert.cvImg2np(im).flat)))

if __name__ == "__main__":
    main()
    