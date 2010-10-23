Sudoku OCR
==========

This is an attempt at implementing the algorithm described [here](http://sudokugrab.blogspot.com/2009/07/how-does-it-all-work.html).
We use the OpenCV Python bindings.

Tested on Windows, with Python 2.6.

Algorithm
---------

Currently the program uses the Nearest Neighbor algorithm to classify the digits. This
means that the program simply stores a lot of labeled images of digits (the
training set). When it needs to classify a new digit, it looks for the stored image
most resembling the digit. It classifies the new digit with the same label
as the stored image. This is not the most intelligent approach but it works.

The author of the article above mentions using Neural Networks for classification.
However, I couldn't get it to work, since the number of input units (pixels in the
digit image) was too big, and the training phase was too slow.

An obvious improvement to try is to reduce the dimensionality of the problem
using e.g. PCA.

Training
--------

You can use the supplied training set. However, if you want to create your
own training set, here's how to do it.

Take some pictures of sudoku puzzles. For each image, create a file called
*imagename*.label which contains a labeling of the digits, for example:

      43  5 8
    5 7 4    
      2  597 
        54 9
     4  3  5
     5 79   2
     1 5  7
        6 8 1
    2 6  83 5

Create a new file containing the filenames of all the images you've used.
This is the fileset. Then run:

    extractFeatures.py fileset.txt > training.txt
    
This will create the training set. Each line in this file contains a label and
a representation of the digit as a vector.

Currently there is no separate training phase -- the prediction program takes a training
set directly.

Testing
-------

Using the training set `vectors.txt`, do

    predict.py vectors.txt puzzle.jpg

and if you're lucky you'll get an ASCII representation of the puzzle.
Question marks will appear in places where the classifier isn't sure.

Limitations
-----------

The algorithm depends on the image resolution in order for the line detection
to work correctly, so you can't use just any image. It works successfully on photos
taken with Nokia 5800, and apparently on iPhone photos as well.

The reason for this limitation is OpenCV's Hough transform implementation. We have to 
use a fixed value for the threshold used to detect lines, and the correct value
depends on the resolution. Even if we choose a correct threshold, the transform
will return several possible lines without indicating their quality. As an improvement
we could use another implementation of the Hough transform which would let
us choose the best lines without knowing a good threshold beforehand.

Dependencies:
-------------

* OpenCV

* numpy

