import numpy as np
import math
from data_row import DataRow

import csv
import os.path


def formatPrice(n):
    if n >= 0:
        curr = "$"
    else:
        curr = "-$"
    return curr + "{0:.2f}".format(abs(n))


def getStockData(key):
    datavec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        datavec.append(float(line.split(",")[4]))

    return datavec


def getFullData(key):
    datavec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        rawData = line.split(",")
        row = DataRow(rawData[0], rawData[1], rawData[2],
                      rawData[3], rawData[4], rawData[5])
        datavec.append(row)

    return datavec


def getState(data, t, window):
    if t - window >= -1:
        vec = data[t - window + 1:t + 1]
    else:
        vec = -(t-window+1)*[data[0]]+data[0: t + 1]
    scaled_state = []
    for i in range(window - 1):
        scaled_state.append(1/(1 + math.exp(vec[i] - vec[i+1])))

    return np.array([scaled_state])


def logTrainingResults(logFile, trainingFile, results, episodeNo):
    # if no log is present yet
    # create copy of training data, and add results
    if(not os.path.exists(logFile)):

        fields = []
        rows = []

        with open(trainingFile, 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            fields = next(csvreader)

            for row in csvreader:
                rows.append(row)

        fields.append(episodeNo)

        rows = addResults(rows, results)

        with open(logFile, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)
    else:
        # log already exists
        # open log and add results
        fields = []
        rows = []

        with open(logFile, 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            fields = next(csvreader)

            for row in csvreader:
                rows.append(row)

        fields.append(episodeNo)

        rows = addResults(rows, results)

        with open(logFile, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)


def addResults(origContent, newContent):
    if len(origContent) != len(newContent):
        return "content size mismatch"
    addedContent = []
    for i in range(len(origContent)):
        oC = origContent[i]
        nC = newContent[i]
        oC.append(nC)
        addedContent.append(oC)
        print("content ", oC)
    return addedContent
