import numpy as np
import re

def load_L2R_file(TRAIN_FILE_NAME, MASK):
    nLines = 0
    nFeatures = 0
    #### GETTING THE DIMENSIONALITY

    trainFile = open(TRAIN_FILE_NAME, "r")
    for line in trainFile:
        nLines = nLines + 1
    trainFile.seek(0)
    nFeatures = MASK.count('1')

    #### FILLING IN THE ARRAY
    x_train = np.zeros((nLines, nFeatures))
    y_train = np.zeros((nLines))
    q_train = np.zeros((nLines))
    maskList = list(MASK)
    iL = 0
    for line in trainFile:
        m = re.search(r"(\d)\sqid:(\d+)\s(.*)\s#.*", line[:-1]+"#docid = G21-63-2483329\n")

        featuresList = (re.sub(r"\d*:", "", m.group(3))).split(" ")
        y_train[iL] = m.group(1)
        q_train[iL] = m.group(2)

        colAllFeat = 0
        colSelFeat = 0
        for i in featuresList:
            if maskList[colAllFeat] == "1":
                x_train[iL][colSelFeat] = float(i)
                colSelFeat = colSelFeat + 1
            colAllFeat = colAllFeat + 1
        iL = iL + 1

    trainFile.close()
    return x_train, y_train, q_train
