import pandas as pd
import matplotlib.pyplot as plt
import string
import math

from itertools import chain, repeat, islice


def getEpisodeColumns(cols):
    episodeColumns = []
    for col in cols:
        if("Episode" in col):
            episodeColumns.append(col)
    return episodeColumns


def reduceList2MaxC(episodes, nth):
    reduced = islice(episodes, nth, None, nth)
    return reduced


def savePlot(file):

    df = pd.read_csv(file, parse_dates=[0], infer_datetime_format=True)

    # plot Close
    plt.subplot(211)
    plt.title("Close")
    plt.plot(df["Date"], df["Close"], '-')
    plt.grid(True)

    # plot results
    plt.subplot(212)
    plt.title("Budget")
    maxLines = 30

    legendColors = []
    # iterating the columns
    for col in reduceList2MaxC(getEpisodeColumns(df.columns), maxLines):
        if("Episode" in col):
            legendColors.append(col)
            plt.plot(df["Date"], df[col], '-')

    # add legend
    plt.legend(legendColors, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file + ".png")
    # plt.show()


#savePlot("training_log/2019-11-15 15:53:47_training.csv")

def createGraph(imageFile, trainingLog, validationLog):

    # Plot Validation records
    with open(validationLog, 'r') as f:
        lines = f.readlines()
        x = [str(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]


    plt.subplot(211)
    plt.title("Validation")
    plt.grid(True)
    plt.plot(x, y)
    plt.xticks(x, [str(i) for i in x], rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=5)

    # Plot Training records
    with open(trainingLog, 'r') as f:
        lines = f.readlines()
        x = [str(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]
 
    plt.subplot(212)
    plt.title("Training")

    plt.plot(x, y)
    plt.xticks(x, [str(i) for i in x], rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.grid(axis='y')

    # show plot
    plt.tight_layout()
    #plt.show()
    plt.savefig(imageFile)


"""
createGraph("graph.png",
            "training_log/2019-11-22_17-03-53/trainingResultsLog.log",
            "training_log/2019-11-22_17-03-53/validationResultsLog.log")
"""