import numpy as np
import math
from data_row import DataRow

import csv
import os.path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

from itertools import chain, repeat, islice


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)

#from plot_log import savePlot


def formatPrice(n):
    if n >= 0:
        curr = "$"
    else:
        curr = "-$"
    return curr + "{0:.2f}".format(abs(n))


def getStockData(file):
    datavec = []
    lines = open(file, "r").read().splitlines()

    for line in lines[1:]:
        datavec.append(float(line.split(",")[4]))

    return datavec


def getFullData(file):
    datavec = []
    lines = open(file, "r").read().splitlines()

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

    # fix length of results by padding with lastKnownResult...
    # necessary for now
    # because the training skips the last day.
    # also if training episode fails due to lack of funds

    lastKnownResult = results[len(results)-1]
    with open(trainingFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        fields = next(csvreader)
        fields = []
        rows = []
        for row in csvreader:
            rows.append(row)

        results = list(pad(results, len(rows), lastKnownResult))

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

        fields.append("Episode 0")

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

        fields.append("Episode {}".format(episodeNo))

        rows = addResults(rows, results)

        with open(logFile, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)

    # savePlot(logFile)


def addResults(origContent, newContent):
    if len(origContent) != len(newContent):
        return "content size mismatch"
    addedContent = []
    for i in range(len(origContent)):
        oC = origContent[i]
        nC = newContent[i]
        oC.append(nC)
        addedContent.append(oC)
        #print("content ", oC)
    return addedContent


def getTimestamp():
    now = datetime.datetime.now()
    time_now = now.strftime("%Y-%m-%d %H:%M:%S")
    return time_now


def assembleFileName(name, format):
    filename = getTimestamp() + "_" + name + format
    return filename


def graph(dates, priceTrend, budgetTrend, episodeNo):

    plt.subplot(211)
    plt.plot(dates, priceTrend, '-')
    plt.xticks([])
    plt.title('Price')

    plt.subplot(212)
    plt.plot(dates, budgetTrend, '-')
    plt.xticks([])
    plt.title('Budget')

    plt.savefig(assembleFileName("Episode_%i" % (episodeNo), ".png"))
    return None
