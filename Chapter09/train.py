from agent import Agent
from helper import getStockData, getState, formatPrice, formatBudget, getFullData, logTrainingResults, logValidationResults, assembleFileName, assembleValidationFileName, createLogDirectory

import numpy as np
from data_row import DataRow
import datetime

import logging

# setup files and names
trainingFile = "data/2008_2016_^GSPC.csv"
validationFile = "data/2017_^GSPC.csv"
testFile = "data/2018_^GSPC Test.csv"

logDirectory = createLogDirectory("training_log")
logFile = logDirectory + assembleFileName("training", ".csv")
log = logDirectory + assembleFileName("log", ".log")


# Gets or creates a logger
logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.DEBUG)

# define file handler and set formatter
file_handler = logging.FileHandler(log)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

window_size = (2, 50)
batch_size = 32
startBudget = 100000
episode_count = 301
validateEvery = 10

agent = Agent(window_size, batch_size, startBudget)

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
    agent.budget = startBudget
    agent.is_eval = False
    done = False
    validationResults = []
    budgetHistory = []
    budgetHistory.append(startBudget)
    state = getState(validation_data, 0, window_size[1] + 1, budgetHistory)
    for t in range(l_validation):
        action = agent.act(state)

        budgetHistory.append(float(formatBudget(agent.budget)))

        next_state = getState(validation_data, t + 1, window_size[1] + 1, budgetHistory)
        reward = 0

        validationDataRow = fullValidationData[t]
        #["Date", "Budget", "Buy/Sell", "Profit", "Reward"]
        validationResult = [validationDataRow.date, "", "", "", 0]

        closePrice = validation_data[t]
        row = fullValidationData[t]

        if action == 1:
            if closePrice < agent.budget:
                agent.inventory.append(closePrice)
                agent.budget -= closePrice
                validationResult[2] = "Buy"
                #logger.debug("Buy: " + formatPrice(closePrice))
            elif(len(agent.inventory) == 0):
                # nothing to sell
                logger.debug("Date: " + row.date)
                logger.debug("Out of budget, terminating validation")
                reward = -999
                done = True
        elif action == 2 and len(agent.inventory) > 0:
            profit = 0
            for bought_price in agent.inventory:
                reward += max(closePrice - bought_price, 0)
                profit += closePrice - bought_price
            agent.inventory = [] # clear inventory, we sold everything
            total_profit += profit
            agent.budget += profit
            validationResult[2] = "Sell"
            validationResult[3] = formatBudget(profit)
            validationResult[4] = reward
            # logger.debug("Sell: " + formatPrice(closePrice) +
            #   " | profit: " + formatPrice(closePrice - bought_price))

        validationResult[1] = formatBudget(agent.budget)
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
    agent.budget = startBudget
    total_profit = 0
    done = False

    dateHistory = []
    budgetHistory = []
    priceHistory = []

    budgetHistory.append(startBudget)

    state = getState(data, 0, window_size[1] + 1, budgetHistory)

    for t in range(l):

        closePrice = data[t]

        row = fullTrainingData[t]

        budgetHistory.append(float(formatBudget(agent.budget)))

        action = agent.act(state)
        action_prob = agent.actor_local.model.predict(state)

        next_state = getState(data, t + 1, window_size[1] + 1, budgetHistory)
        reward = 0

        if action == 1:
            if closePrice < agent.budget:
                agent.inventory.append(closePrice)
                agent.budget-=closePrice
                # logger.debug("Buy. Price is " + formatPrice(closePrice)
                #logger.debug("Buget is: " + formatPrice(agent.budget))
                #logger.debug("day " + row.date)
            else:
                """
                logger.debug("Date: " + row.date)
                logger.debug("Can not buy.")
                logger.debug("Price is: " + formatPrice(closePrice))
                logger.debug("Buget is: " + formatPrice(agent.budget))
                """
                # nothing to sell
                if(len(agent.inventory) == 0):
                    logger.debug("Date: " + row.date)
                    logger.debug("Out of budget, terminating episode")
                    failedEpisodes.append(e)
                    reward = -999
                    done = True

        elif action == 2 and len(agent.inventory) > 0:
            profit = 0
            for bought_price in agent.inventory:
                reward += max(closePrice - bought_price, 0)
                profit += closePrice - bought_price
            agent.inventory = [] # clear inventory, we sold everything
            total_profit += profit
            agent.budget += profit
            #logger.debug("sell: " + formatPrice(closePrice) + "| profit: " + formatPrice(data[t] - bought_price))

        if t == l - 1:
            done = True
        agent.step(action_prob, reward, next_state, done)
        state = next_state

        if done:
            logger.debug("------------------------------------------")
            logger.debug("Total Profit: " + formatPrice(total_profit))
            logger.debug("------------------------------------------")
            break  # break out of training episode

    #graph(dateHistory, priceHistory, budgetHistory, e)
    logTrainingResults(logFile, trainingFile, budgetHistory, e)
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
    logger.debug("skipping TEST")
    import sys
    sys.exit()

test_data = getStockData(testFile)
l_test = len(test_data) - 1
state = getState(test_data, 0, window_size[1] + 1)
total_profit = 0
agent.inventory = []
agent.budget = startBudget
agent.is_eval = False
done = False
for t in range(l_test):
    action = agent.act(state)

    next_state = getState(test_data, t + 1, window_size[1] + 1)
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
