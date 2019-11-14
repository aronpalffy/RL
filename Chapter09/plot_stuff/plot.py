from pandas import read_csv
from matplotlib import pyplot


series = read_csv("data/" + "^GSPC Test" + ".csv", 
header=0,
index_col=0, 
parse_dates=True, 
squeeze=False,
usecols=[0,4])
series.plot()
pyplot.show()