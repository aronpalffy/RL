import datetime
import logging

import numpy as np

from agent import Agent
from data_row import DataRow
from helper import (assembleFileName, assembleValidationFileName,
                    createLogDirectory, formatNumber, formatPrice, getFullData,
                    getState, getStockData, logTrainingResults,
                    logValidationResults)
from plot_log import createGraph

# setup files and names
#trainingFile = "data/orig_training_14-08-2006-13-08-2015^GSPC.csv"
trainingFile = "data/orig_training_14-08-2006-13-08-2015^GSPC.csv"
validationFile = "data/orig_test_14-08-2015-14-08-2018^GSPC.csv"
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
    
    handler = logging.FileHandler(log_file)
    # no formatter for these logs
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
logger.addHandler(logging.StreamHandler())

""" PARAMETERS """

LR_ACTOR = 0.0001 # learning rate of actor
LR_CRITIC = 0.001 # learning rate of critic
# fuzz factor
OPTIMIZER_EPSILON = 0.001  # K.epsion() defaults to 1e-07 (0.0000001)

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999995
"""
self.epsilon_decay = 0.999999 with this setting
epsilon decays to 0.5057234296206272 in 301 epizodes of training
on orig_training_14-08-2006-13-08-2015^GSPC 
"""
"""
self.epsilon_decay = 0.99999 with this setting
epsilon decays to 0.009999971601096055 in 301 epizodes of training
on orig_training_14-08-2006-13-08-2015^GSPC 
"""
IS_EVAL = False

window_size = 50
batch_size = 32
episode_count = 301
validateEvery = 10

agent = Agent(window_size, batch_size, IS_EVAL, EPSILON, EPSILON_MIN,
              EPSILON_DECAY, LR_ACTOR, LR_CRITIC, OPTIMIZER_EPSILON)

logger.debug("The following paramters are used for this training")
logger.debug("window_size = {}".format(window_size))
logger.debug("batch_size = {}".format(batch_size))
logger.debug("EPSILON = {}".format(EPSILON))
logger.debug("EPSILON_MIN = {}".format(EPSILON_MIN))
logger.debug("EPSILON_DECAY = {}".format(EPSILON_DECAY))
logger.debug("LR_ACTOR = {}".format(LR_ACTOR))
logger.debug("LR_CRITIC = {}".format(LR_CRITIC))
logger.debug("OPTIMIZER_EPSILON = {}".format(OPTIMIZER_EPSILON))

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
    agent.is_eval = True
    done = False
    validationResults = []

    state = getState(validation_data, 0, window_size + 1)
    for t in range(l_validation):
        action = agent.act(state)
        # TODO check if we need this?!
        #action_prob = agent.actor_target.model.predict(state)

        next_state = getState(validation_data, t + 1,
                              window_size + 1)
        reward = 0

        validationDataRow = fullValidationData[t]
        #["Date", "Buy/Sell", "Profit", "Reward"]
        validationResult = [validationDataRow.date, "", "", 0]

        closePrice = validation_data[t]
        #row = fullValidationData[t]

        if action == 1:
            agent.inventory.append(closePrice)
            validationResult[1] = "Buy"

        elif action == 2 and len(agent.inventory) > 0:

            bought_price = agent.inventory.pop(0)
            profit = closePrice - bought_price
            reward = max(profit, 0)
            total_profit += profit
            validationResult[1] = "Sell"
            validationResult[2] = formatNumber(profit)
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

    validationRecord.info("V_{}    {}".format(
        f'{episodeNo:03}', formatNumber(total_profit)))


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

    actionMatchesPrediction = []

    agent.inventory = []
    agent.is_eval = False
    total_profit = 0
    done = False

    state = getState(data, 0, window_size + 1)

    for t in range(l):

        closePrice = data[t]

        row = fullTrainingData[t]

        action = agent.act(state)
        action_prob = agent.actor_local.model.predict(state)
        """
        action_prob_actor_target = agent.actor_target.model.predict(state)
        action_prob_critic_local = agent.critic_local.model.predict(state)
        action_prob_critic_target = agent.critic_target.model.predict(state)
        """
        maximum = np.max(action_prob)
        index_of_maximum = np.argmax(action_prob)
        if action == index_of_maximum:
            actionMatchesPrediction.append("{} {}".format(row.date, action))

        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:
            agent.inventory.append(closePrice)
            # logger.debug("Buy. Price is " + formatPrice(closePrice)
            #logger.debug("Buget is: " + formatPrice(agent.budget))
            #logger.debug("day " + row.date)

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            profit = closePrice - bought_price
            reward = max(profit, 0)
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
    matches = len(actionMatchesPrediction)
    percent = matches/l*100
    logger.debug(
        "actions matching predicions {}/{} that is {}%".format(matches, l,
        formatNumber(percent)))
    logger.debug("current value of agent.epsilon: {}".format(agent.epsilon))
    logger.debug("")

    #if e % validateEvery == 0:
    #validate(e)
    durations.append(episodeDuration.total_seconds())
    trainingRecord.info("T_{}    {}    {}".format(
        f'{e:03}', formatNumber(total_profit),formatNumber(percent)))

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
        profit = test_data[t] - bought_price
        reward = max(profit, 0)
        total_profit += profit
        logger.debug("Sell: " + formatPrice(test_data[t]) +
                     " | profit: " + formatPrice(profit))

    if t == l_test - 1:
        done = True
    agent.step(action_prob, reward, next_state, done)
    state = next_state

    if done:
        logger.debug("------------------------------------------")
        logger.debug("Total Profit: " + formatPrice(total_profit))
        logger.debug("------------------------------------------")
