import os
from plot_log import createGraph

directory = 'training_log'
latest_log = str(max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime))

createGraph(latest_log+"/graph.png",
            latest_log+"/trainingResultsLog.log",
            latest_log+"/validationResultsLog.log"
            )