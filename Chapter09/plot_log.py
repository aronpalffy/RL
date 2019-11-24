import pandas as pd
import matplotlib.pyplot as plt
import string
import math
import numpy as np

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
    fig, axs = plt.subplots(2,1)

    # Plot Validation records
    with open(validationLog, 'r') as f:
        lines = f.readlines()
        x = [str(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]

    #plt.subplot(211)
    plt.title("Validation")
    axs[0].grid(axis='y')
    axs[0].plot(x, y)
    #axs[0].xticks(x, [str(i) for i in x], rotation=90)
    axs[0].tick_params(axis='x', which='major', labelsize=1)

    # Plot Training records
    with open(trainingLog, 'r') as f:
        lines = f.readlines()
        x = [str(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]
        percentMatches = [float(line.split()[2]) for line in lines]

    #plt.subplot(212)
    plt.title("Training")
    axs[1].plot(x, y)
    color = 'tab:blue'
    axs[1].set_xlabel('episodes')
    axs[1].set_ylabel('profit', color=color)
    axs[1].tick_params(axis='y', labelcolor=color)
    axs[1].tick_params(axis='x', labelsize=5)

    k = 10
    ys = x[::k]
    ys = np.append(ys, x[-1])
    labels = x[::k]
    labels = np.append(labels, x[-1])
    plt.xticks(ys, labels, rotation=90)
    
    color = 'tab:red'
    ax2 = axs[1].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, percentMatches, lw=0.5, color=color)
    ax2.set_ylabel('% match', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(axis='x', labelsize=5)

    axs[1].grid(axis='y')
    k = 10
    ys = x[::k]
    ys = np.append(ys, x[-1])
    labels = x[::k]
    labels = np.append(labels, x[-1])
    plt.xticks(ys, labels, rotation=90)

    # show plot
    fig.tight_layout()
    #plt.show()
    plt.savefig(imageFile)


"""
createGraph("graph.png",
            "training_log/2019-11-24_11-35-08/trainingResultsLog.log",
            "training_log/2019-11-24_11-35-08/validationResultsLog.log")
"""
