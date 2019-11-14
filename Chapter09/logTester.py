from helper import logTrainingResults

logFile = "training_log/newLog.csv"

trainingFile = "training_log/dummy.csv"

results = [111, 222, 333, 444, 555]

episodeNo = 1

logTrainingResults(logFile, trainingFile, results, episodeNo)
