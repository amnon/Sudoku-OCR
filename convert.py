import cv
import numpy as np

# function for conversion between Numpy and OpenCV,
# since OpenCV2.1 default build doesn't come with the documented cv.fromarray
# function

############################
def cvImg2np(cvImg):
    """
    Convert OpenCV image to numpy array.
    Usage: cvImg2np(cvImg) --> npMat
"""
    width = cvImg.width
    height = cvImg.height
    nchannels = cvImg.nChannels
    ##############
    assert nchannels == 1 or nchannels == 3 or nchannels == 4,\
        'Failed: nchannels != 1 or nchannels != 3 or nchannels != 4'

    assert cvImg.depth == cv.IPL_DEPTH_8U or\
        cvImg.depth == cv.IPL_DEPTH_32F or\
        cvImg.depth == cv.IPL_DEPTH_64F,\
        'wrong bit depth'

    if nchannels == 1:
        if cvImg.depth == cv.IPL_DEPTH_8U:
            np_array = np.fromstring(cvImg.tostring(), dtype=np.uint8)
        if cvImg.depth == cv.IPL_DEPTH_32F:
            np_array = np.fromstring(cvImg.tostring(), dtype=np.float32)
        if cvImg.depth == cv.IPL_DEPTH_64F:
            np_array = np.fromstring(cvImg.tostring(), dtype=np.float64)

        np_mat = np.reshape(np_array, (height,width))
        return np_mat

    if nchannels >= 3:
        if cvImg.depth == cv.IPL_DEPTH_8U:
            np_array = np.fromstring(mat.tostring(), dtype=np.unit8)

        if cvImg.depth == cv.IPL_DEPTH_32F:
            np_array = np.fromstring(mat.tostring(), dtype=np.float32)

        if cvImg.depth == cv.IPL_DEPTH_64F:
            np_array = np.fromstring(mat.tostring(), dtype=np.float64)

        np_mat = np.reshape(np_array, (height,width,nChannels))
        return np_mat

def cvMat2np(cvMat, npType, cvType):
    """
    Convert OpenCV matrix to numpy array.
    Usage: cvMat2np(cvMat, npType = np.float32) --> npMat
    input should be cvMat either of type 32F, 64F, 32S, or 64S
"""

    rows = cvMat.rows
    cols = cvMat.cols
    step = cvMat.step
    cvType = cvMat.type

    assert cvType != cv.CV_8UC1 and cvType != cv.CV_8UC2 and cvType != cv.CV_8UC3 and\
        cvType != cv.CV_8SC1 and cvType != cv.CV_8SC2 and cvType != cv.CV_8SC3 and\
        cvType != cv.CV_16UC1 and cvType != cv.CV_16UC2 and cvType != cv.CV_16UC3 and\
        cvType != cv.CV_16SC1 and cvType != cv.CV_16SC2 and cvType != cv.CV_16SC3, "CvMat type error"

    if npType == np.float32 or npType == np.int32:

        nchannels = step/(cols*4)

    elif npType == np.float64 or npType == np.int64:

        nchannels = step/(cols*8)

    else:

        assert False, "This nptype is not allowed for this conversion"


    if nchannels == 1:

        np_array = np.fromstring(cvMat.tostring(), dtype=npType)
        np_mat = np.reshape(np_array, (rows,cols))
        return np_mat

    elif nchannels >= 2:

        np_array = np.fromstring(cvMat.tostring(), dtype=npType)
        np_mat = np.reshape(np_array, (rows,cols,nchannels))
        return np_mat

def np2cvImg(npMat, npType, cvType):
    """
    np2cvImg(npMat, npType=np.float32) --> IplImage

        input should be np.float32 of channel one or three
"""
    height = npMat.shape[0]
    width = npMat.shape[1]
    try:

        nchannels = npMat.shape[2]

    except:

        nchannels = 1

    assert nchannels == 1 or nchannels == 3 or nchannels == 4, 'wrong number of channels'

    if npType == np.float32:

        cvMat = cv.CreateImageHeader((width, height), cv.IPL_DEPTH_32F)
        cv.SetData(npMat, arr.tostring(), width*4*nchannels)
        return cvMat

    else:

        assert False, 'wrong numpy type'


def np2cvMat(npMat):
    """
    input must be either np.float32, float64, or int32
"""
    rows = npMat.shape[0]
    cols = npMat.shape[1]
    try:
        nchannels = npMat.shape[2]
    except:
        nchannels = 1

    npType = npMat.dtype
    
    if npType == np.float32:
        cvType = cv.CV_32FC1
        size = 4
    elif npType == np.float64:
        cvType = cv.CV_64FC1
        size = 8
    elif npType == np.int32:
        cvType = cv.CV_32SC1
        size = 4
    else:
        assert False, 'wrong type'
    
    cvMat = cv.CreateMatHeader(rows, cols, cvType)
    cv.SetData(cvMat, npMat.tostring(), cols*size*nchannels)
    return cvMat
