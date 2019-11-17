from agent import Agent
from helper import getStockData, getState, formatPrice, getFullData, logTrainingResults, assembleFileName

import numpy as np
from data_row import DataRow
import datetime

window_size = (50, 4)
batch_size = 32
startBudget = 10000
episode_count = 300

agent = Agent(window_size, batch_size, startBudget)

trainingFile = "data/^GSPC.csv"
testFile = "data/^GSPC Test.csv"
logFile = "training_log/" + assembleFileName("training", ".csv")

data = getStockData(trainingFile)
fullData = getFullData(trainingFile)
l = len(data) - 1

durations = []
failedEpisodes = []
trainingStart = datetime.datetime.now()
for e in range(episode_count):
    episodeStart = datetime.datetime.now()
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size[0] + 1)

    agent.inventory = []
    agent.budget = startBudget
    total_profit = 0
    done = False

    dateHistory = []
    budgetHistory = []
    priceHistory = []

    for t in range(l):

        closePrice = data[t][3]

        row = fullData[t]

        dateHistory.append(row.date)
        budgetHistory.append(agent.budget)
        priceHistory.append(closePrice)
        """
        plt.subplot(211)  
        plt.plot(row.date, agent.budget,
        color='blue', marker='o',
        linestyle='dashed',
        linewidth=2, markersize=1)
        plt.subplot(212)  
        plt.plot(row.date, closePrice,
        color='red', marker='+',
        linestyle='dashed',
        linewidth=2, markersize=1)
        plt.pause(0.05)
        """
        action = agent.act(state)
        action_prob = agent.actor_local.model.predict(state)

        next_state = getState(data, t + 1, window_size[0] + 1)
        reward = 0

        if action == 1:
            if closePrice < agent.budget:
                agent.inventory.append(closePrice)
                # print("Buy. Price is " + formatPrice(closePrice)
                #print("Buget is: " + formatPrice(agent.budget))
                #print("day " + row.date)
            else:
                """
                print("Date: " + row.date)
                print("Can not buy.")
                print("Price is: " + formatPrice(closePrice))
                print("Buget is: " + formatPrice(agent.budget))
                """
                # nothing to sell
                if(len(agent.inventory) == 0):
                    print("Date: " + row.date)
                    print("Out of budget, terminating episode")
                    failedEpisodes.append(e)
                    done = True

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(closePrice - bought_price, 0)
            total_profit += closePrice - bought_price
            agent.budget += closePrice - bought_price
            #print("sell: " + formatPrice(closePrice) + "| profit: " + formatPrice(data[t] - bought_price))

        if t == l - 1:
            done = True
        agent.step(action_prob, reward, next_state, done)
        state = next_state

        if done:
            print("------------------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("------------------------------------------")
            break  # break out of training episode

    #graph(dateHistory, priceHistory, budgetHistory, e)
    logTrainingResults(logFile, trainingFile, budgetHistory, e)
    episodeEnd = datetime.datetime.now()
    duration = episodeEnd-episodeStart
    print(duration.total_seconds())
    durations.append(duration.total_seconds())
if True:
    avgDuration = sum(durations) / len(durations)
    print("Avg episode Duration: ", avgDuration)
    trainingEnd = datetime.datetime.now()
    trainingDuration = trainingEnd-trainingStart
    print("Training duration: ", trainingDuration)
    print("{0} episodes failed out of {1}".format(len(failedEpisodes), episode_count))
    print(failedEpisodes)
    print("skipping TEST")
    import sys
    sys.exit()

test_data = getStockData(testFile)
l_test = len(test_data) - 1
state = getState(test_data, 0, window_size[0] + 1)
total_profit = 0
agent.inventory = []
agent.budget = startBudget
agent.is_eval = False
done = False
for t in range(l_test):
    action = agent.act(state)

    next_state = getState(test_data, t + 1, window_size[0] + 1)
    reward = 0

    if action == 1:

        agent.inventory.append(test_data[t])
        print("Buy: " + formatPrice(test_data[t]))

    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(test_data[t] - bought_price, 0)
        total_profit += test_data[t] - bought_price
        print("Sell: " + formatPrice(test_data[t]) +
              " | profit: " + formatPrice(test_data[t] - bought_price))

    if t == l_test - 1:
        done = True
    agent.step(action_prob, reward, next_state, done)
    state = next_state

    if done:
        print("------------------------------------------")
        print("Total Profit: " + formatPrice(total_profit))
        print("------------------------------------------")
