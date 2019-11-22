from agent import Agent
from helper import getStockData, getState, formatPrice, formatBudget, getFullData, logTrainingResults, logValidationResults, assembleFileName, assembleValidationFileName, createLogDirectory

import numpy as np
from data_row import DataRow
import datetime

from plot_log import createGraph

import logging

# setup files and names
trainingFile = "data/2008_2016_^GSPC.csv"
validationFile = "data/2017_^GSPC.csv"
testFile = "data/2018_^GSPC Test.csv"

logDirectory = createLogDirectory("training_log")
logFile = logDirectory + assembleFileName("training", ".csv")
imageFile = logDirectory + assembleFileName("graph", ".png")

# prepare logfile names
log = logDirectory + assembleFileName("log", ".log")
validationResultsLogFile = logDirectory + \
    assembleFileName("validationResultsLog", ".log")
trainingResultsLogFile = logDirectory + \
    assembleFileName("trainingResultsLog", ".log")


def setup_logger(name, log_file, level=logging.INFO):
    """unctin setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    # handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# Gets or creates a logger
logger = logging.getLogger(__name__)
validationRecord = setup_logger(
    "ValidationResults", validationResultsLogFile, logging.INFO)
trainingRecord = setup_logger(
    "TrainnResults", trainingResultsLogFile, logging.INFO)
# set log level
logger.setLevel(logging.DEBUG)

# define file handler and set formatter
file_handler = logging.FileHandler(log)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

window_size = 50
batch_size = 32
episode_count = 11
validateEvery = 10

agent = Agent(window_size, batch_size)

data = getStockData(trainingFile)
fullTrainingData = getFullData(trainingFile)

validation_data = getStockData(validationFile)
fullValidationData = getFullData(validationFile)

# VALIDATION


def validate(episodeNo):
    validationLogFile = assembleValidationFileName(logFile, episodeNo, ".csv")
    logger.debug("*******************")
    logger.debug("starting validation after episode {}".format(episodeNo))
    #
    # TODO save model
    #
    l_validation = len(validation_data) - 1
    total_profit = 0
    agent.inventory = []
    agent.is_eval = False
    done = False
    validationResults = []

    state = getState(validation_data, 0, window_size + 1)
    for t in range(l_validation):
        action = agent.act(state)

        next_state = getState(validation_data, t + 1,
                              window_size + 1)
        reward = 0

        validationDataRow = fullValidationData[t]
        #["Date", "Buy/Sell", "Profit", "Reward"]
        validationResult = [validationDataRow.date, "", "", 0]

        closePrice = validation_data[t]
        row = fullValidationData[t]

        if action == 1:
            agent.inventory.append(closePrice)
            validationResult[1] = "Buy"

        elif action == 2 and len(agent.inventory) > 0:

            bought_price = agent.inventory.pop(0)
            reward = max(closePrice - bought_price, 0)
            profit = closePrice - bought_price
            total_profit += profit
            validationResult[1] = "Sell"
            validationResult[2] = formatBudget(profit)
            validationResult[3] = reward
            # logger.debug("Sell: " + formatPrice(closePrice) +
            #   " | profit: " + formatPrice(closePrice - bought_price))

        validationResults.append(validationResult)
        if t == l_validation - 1:
            done = True
        agent.step(action_prob, reward, next_state, done)
        state = next_state

        if done:
            logValidationResults(validationLogFile, validationResults)
            logger.debug("------------------------------------------")
            logger.debug("Total Profit of Validation: " +
                         formatPrice(total_profit))
            logger.debug("------------------------------------------")
            break  # break out of validation

    validationRecord.info("Validation_{}    {}".format(
        episodeNo, formatBudget(total_profit)))

# TRAINING
durations = []
failedEpisodes = []
l = len(data) - 1

# begin training
trainingStart = datetime.datetime.now()
for e in range(episode_count):
    episodeStart = datetime.datetime.now()
    logger.debug("*******************")
    logger.debug("Episode " + str(e) + "/" + str(episode_count))

    agent.inventory = []
    total_profit = 0
    done = False

    state = getState(data, 0, window_size + 1)

    for t in range(l):

        closePrice = data[t]

        row = fullTrainingData[t]

        action = agent.act(state)
        action_prob = agent.actor_local.model.predict(state)

        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:
            agent.inventory.append(closePrice)
            # logger.debug("Buy. Price is " + formatPrice(closePrice)
            #logger.debug("Buget is: " + formatPrice(agent.budget))
            #logger.debug("day " + row.date)

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(closePrice - bought_price, 0)
            profit = closePrice - bought_price
            total_profit += profit
            #logger.debug("sell: " + formatPrice(closePrice) + "| profit: " + formatPrice(data[t] - bought_price))

        if t == l - 1:
            done = True
        agent.step(action_prob, reward, next_state, done)
        state = next_state

        if done:
            logger.debug("------------------------------------------")
            logger.debug("Total Profit: " + formatPrice(total_profit))
            logger.debug("------------------------------------------")
            # break  # break out of training episode

    #graph(dateHistory, priceHistory, budgetHistory, e)
    #logTrainingResults(logFile, trainingFile, e)
    episodeEnd = datetime.datetime.now()
    episodeDuration = episodeEnd-episodeStart
    trainingDuration = episodeEnd-trainingStart

    logger.debug("episodeDuration {}".format(episodeDuration.total_seconds()))
    logger.debug("trainingDuration {}".format(
        trainingDuration))
    logger.debug("")

    if e % validateEvery == 0:
        validate(e)
    durations.append(episodeDuration.total_seconds())
    trainingRecord.info("Training_{}    {}".format(
        e, formatBudget(total_profit)))

if True:
    avgDuration = sum(durations) / len(durations)
    logger.debug("Avg episode Duration: {}".format(
        str(avgDuration)))
    trainingEnd = datetime.datetime.now()
    totalTrainingDuration = trainingEnd-trainingStart
    logger.debug("Total training duration: {}".format(
        str(totalTrainingDuration)))
    logger.debug("{0} episodes failed out of {1}".format(
        len(failedEpisodes), episode_count))
    # logger.debug(failedEpisodes)
    logger.debug("creating graph")
    createGraph(imageFile, trainingResultsLogFile, validationResultsLogFile)
    logger.debug("skipping TEST")
    import sys
    sys.exit()

test_data = getStockData(testFile)
l_test = len(test_data) - 1
state = getState(test_data, 0, window_size + 1)
total_profit = 0
agent.inventory = []
agent.is_eval = False
done = False
for t in range(l_test):
    action = agent.act(state)

    next_state = getState(test_data, t + 1, window_size + 1)
    reward = 0

    if action == 1:

        agent.inventory.append(test_data[t])
        logger.debug("Buy: " + formatPrice(test_data[t]))

    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(test_data[t] - bought_price, 0)
        total_profit += test_data[t] - bought_price
        logger.debug("Sell: " + formatPrice(test_data[t]) +
                     " | profit: " + formatPrice(test_data[t] - bought_price))

    if t == l_test - 1:
        done = True
    agent.step(action_prob, reward, next_state, done)
    state = next_state

    if done:
        logger.debug("------------------------------------------")
        logger.debug("Total Profit: " + formatPrice(total_profit))
        logger.debug("------------------------------------------")
