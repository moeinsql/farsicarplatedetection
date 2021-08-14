import numpy as np
import os
import cv2

def loadKNNDataAndTrainKNN():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly
    kNearest = cv2.ml.KNearest_create()

    try:
        npaClassifications = np.loadtxt(os.path.join(os.path.dirname(__file__), "knnmodel/classifications.txt"), np.float32)                  # read in training classifications
    except:                                                                                 # if file could not be opened
        print("error, unable to open classifications.txt, exiting program\n")  # show error message
        #os.system("pause")
        return False                                                                        # and return False
    # end try

    try:
        npaFlattenedImages = np.loadtxt(os.path.join(os.path.dirname(__file__), "knnmodel/flattened_images.txt"), np.float32)                 # read in training images
    except:                                                                                 # if file could not be opened
        print("error, unable to open flattened_images.txt, exiting program\n")  # show error message
        #os.system("pause")
        return False                                                                        # and return False
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest.setDefaultK(1)                                                             # set default K to 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # train KNN object

    return True, kNearest