import mytry
import sys
import numpy as np
import convert

# nearest neighbor classifier (also known as K-Nearest Neighbor, but
# currently K must be 1)
class KNNClassifier:
    def __init__(self, fname):
        self.labels = []
        self.samples = []
        for line in open(fname):
            label, sample = line.rstrip().split(",", 1)
            self.labels.append(label)
            sample = [int(x) for x in sample.split(",")]
            self.samples.append(sample)
        self.samples = np.array(self.samples)

    def classify(self, vec):
        # compute (squared) distances between vector and each member of training set,
        # and choose label of nearest neighbor
        dists2 = np.sum((vec-self.samples)**2, 1)
        nearest = np.argmin(dists2)
        # since vectors are binary, dists2 is actually Hamming distance.
        # don't return result if more than a certain number of the bits are different.
        return self.labels[nearest] if dists2[nearest] < sum(vec)/2 else '?'
        
def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s trainingSet puzzle\n" % sys.argv[0])
        sys.exit(1)
    classifier = KNNClassifier(sys.argv[1])
    digits = mytry.processImage(sys.argv[2], interactive=False)
    res = [[' ' for i in range(9)] for j in range(9)]
    for ((i,j), im) in digits:
        # convert to same format as used in extractFeatures
        # TODO: this should already be returned from processImage
        im = list((convert.cvImg2np(im)/255).flat)
    
        label = classifier.classify(im)
        res[j][i] = label
        
            
    for j, line in enumerate(res):
        if j > 0 and j % 3 == 0:
            sys.stdout.write("------+------+------\n")
        for i, label in enumerate(line):
            if i > 0 and i % 3 == 0:
                sys.stdout.write("|")
            sys.stdout.write(" " + label)
        sys.stdout.write("\n")

if __name__ == "__main__":
    main()