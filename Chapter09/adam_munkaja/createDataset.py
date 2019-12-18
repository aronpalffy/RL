import csv
import datetime
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from scipy import stats

import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR

thisdict = {}

file = open('Chapter09/adam_munkaja/GOLD.csv')
reader = csv.reader(file, delimiter=',')
for row in reader:
    # print(row[0])
    # print(row[:11].lstrip())
    # date_time_str = '2018-06-29 08:15:27.243860'
    # date_time_obj = datetime.datetime.strptime(row[0], '%Y-%m-%d')
    date_time_obj = datetime.datetime.strptime(row[0], '%Y-%m-%d')
    # # for column in row:
    if date_time_obj in thisdict:
        thisdict[date_time_obj] = thisdict.get(date_time_obj) + "; " + row[4]
    else:
        thisdict[date_time_obj] = row[4]

dfGold = pd.DataFrame.from_dict(thisdict, orient='index').astype(float)
thisdict = {}

file = open('Chapter09/adam_munkaja/Silver.csv')
reader = csv.reader(file, delimiter=';')
for row in reader:
    # print(row[0].split(", ")[0])
    # print(row[:11].lstrip())
    # date_time_str = '2018-06-29 08:15:27.243860'
    # date_time_obj = datetime.datetime.strptime(row[0], '%Y-%m-%d')
    date_time_obj = datetime.datetime.strptime(row[0].split(", ")[0].strip(), '%m %d %Y')
    # for column in row:
    if date_time_obj in thisdict:
        thisdict[date_time_obj] = thisdict.get(date_time_obj) + "; " + row[0].split(", ")[1]
    else:
        thisdict[date_time_obj] = row[0].split(", ")[1]

dfSilver = pd.DataFrame.from_dict(thisdict, orient='index').astype(float)
thisdict = {}

file = open('Chapter09/adam_munkaja/Oil_Filtered.csv')
reader = csv.reader(file, delimiter=',')
for row in reader:
    date_time_obj = datetime.datetime.strptime(row[0], '%Y-%m-%d')
    if date_time_obj in thisdict:
        thisdict[date_time_obj] = thisdict.get(date_time_obj) + "; " + row[1]
    else:
        thisdict[date_time_obj] = row[1]

dfOil = pd.DataFrame.from_dict(thisdict, orient='index').replace(to_replace =".",
                 value ="0").astype(float)
thisdict = {}

file = open('Chapter09/adam_munkaja/USD_EUR.csv')
reader = csv.reader(file, delimiter=';')
for row in reader:
    # print(row[0].split(", ")[0])
    # print(row[:11].lstrip())
    # date_time_str = '2018-06-29 08:15:27.243860'
    usderurrow = row[0].split(",")
    # print(usderurrow)
    price = usderurrow[2].replace('"', '')
    # print(price)
    # row = row.replace('ď»ż', '').replace('Â·', '.')
    datestring = (usderurrow[0] + usderurrow[1]).replace('ď»ż"', '').replace('Â·', '.').replace('"', '').replace('\ufeff', '')
    #print(datestring)
    date_time_obj = datetime.datetime.strptime(datestring, '%m %d %Y')
    # for column in row:
    if date_time_obj in thisdict:
        thisdict[date_time_obj] = thisdict.get(date_time_obj) + "; " + price
    else:
        thisdict[date_time_obj] = price

dfUsdEur = pd.DataFrame.from_dict(thisdict, orient='index').astype(float)
thisdict = {}

file = open('Chapter09/adam_munkaja/Rate.csv')
reader = csv.reader(file, delimiter=',')
for row in reader:
    row = row[0].replace('ď»ż', '').replace('Â·', '.').replace('\ufeff', '').replace('·', '.')
    #date_time_obj = datetime.datetime.strptime(row[:11].lstrip(), '%Y %m %d')
    date_time_obj = datetime.datetime.strptime(row[:11].lstrip(), '%Y %m %d')
    #print(date_time_obj)
    if date_time_obj in thisdict:
        thisdict[date_time_obj] = thisdict.get(date_time_obj) + "; " + row[-4:].lstrip()
    else:
        thisdict[date_time_obj] = row[-4:].lstrip()

dfRate = pd.DataFrame.from_dict(thisdict, orient='index').astype(float)
thisdict = {}

file = open('Chapter09/adam_munkaja/GSPC_X.csv')
reader = csv.reader(file, delimiter=',')
for row in reader:
    # print(row[0])
    # print(row[:11].lstrip())
    # date_time_str = '2018-06-29 08:15:27.243860'
    # date_time_obj = datetime.datetime.strptime(row[0], '%Y-%m-%d')
    date_time_obj = datetime.datetime.strptime(row[0], '%Y-%m-%d')
    # # for column in row:
    if date_time_obj in thisdict:
        thisdict[date_time_obj] = thisdict.get(date_time_obj) + "; " + row[4]
    else:
        thisdict[date_time_obj] = row[4]

    # print(date_time_obj)
# print(thisdict)

dfStock = pd.DataFrame.from_dict(thisdict, orient='index').astype(float)
thisdict = {}

dataset = pd.concat([dfGold, dfSilver, dfUsdEur, dfOil , dfRate, dfStock], axis=1)


dataset = dataset.loc['2000-01-03' : '2019-12-09']
# TODO too much fill
dataset = dataset.fillna(method='pad')
dataset = dataset.dropna()

dataset.columns = ["Gold", "Silver", "UsdEur", "Oil", "Rate", "Stock"]

print(dataset.head())
print(dataset.isnull().sum())

# dataset = dataset.T

register_matplotlib_converters()
# Plot
fig, axes = plt.subplots(nrows=3, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
     data = dataset[dataset.columns[i]]
     ax.plot(data, color='red', linewidth=1)
     # Decorations
     ax.set_title(dataset.columns[i])
     ax.xaxis.set_ticks_position('none')
     ax.yaxis.set_ticks_position('none')
     ax.spines["top"].set_alpha(0)
     ax.tick_params(labelsize=6)
plt.tight_layout()
#plt.show()
plt.savefig("Chapter09/adam_munkaja/data.png")



# dataset = pplt.show()d.DataFrame.from_dict(thisdict)
# dataset = pd.Series(thisdict).to_frame()
# dataset.head()

# with open('data_file.csv', mode='w') as data_file:
#     employee_writer = csv.writer(employee_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#
#     employee_writer.writerow(['John Smith', 'Accounting', 'November'])
#     employee_writer.writerow(['Erica Meyers', 'IT', 'March'])

# csv_columns = ['Gold', 'Silver', 'Country', 'Oil', 'USD_EUR', 'Rate', 'Stock']
# csv_file = "data_file.csv"
# try:
#     with open(csv_file, 'w', newline='') as csvfile:
#         # writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#         writer = csv.writer(csvfile, delimiter=';')
#         # writer.writeheader()
#         # writer.writeheader()
#         for key in thisdict.keys():
#             listrow = []
#             datastr = key.strftime("%Y-%m-%d")
#             listrow.append(datastr)
#             print(thisdict[key].split(", "))
#
#             listrow.append(thisdict[key].split(", "))
#             # writer.writerow(datastr + ";" + thisdict[key])
#             writer.writerow(listrow)
#         #
#         # for key, value in thisdict.items():
#         #     print(key, '->', value)
#         #     datastr = key.strftime("%Y-%m-%d")
#         #     writer.writerow(value)
# except IOError:
#     print("I/O error")

# plt.figure(figsize=(14,6))
# plt.subplot(1,2,1)
# dataset['Gold'].hist(bins=50)
# plt.title('Gold')
# plt.subplot(1,2,2)
# plt.show()
#
# stats.probplot(dataset['Gold'], plot=plt);
# dataset.Gold.describe().T
# plt.show()

nobs = 15
X_train, X_test = dataset[0:-nobs], dataset[-nobs:]
print(X_train.shape)
print(X_test.shape)

transform_data = X_train.diff().dropna()
print(transform_data.head())
print(transform_data.describe())

# TODO Stationarity check
# TODO Stationarity check
# TODO Stationarity check

mod = smt.VAR(transform_data)
res = mod.fit(maxlags=15, ic='aic')
print(res.summary())

# TODO Durbin-Watson Statistic

pred = res.forecast(transform_data.values[3:], 15)
pred_df = pd.DataFrame(pred, index=dataset.index[-15:], columns=dataset.columns)
print(pred_df)

pred_inverse = pred_df.cumsum()
f = pred_inverse + X_test#.shift(1)
print(f)
#
# pred_inverse = pred_df.cumsum()
# f = pred_inverse.shift(-1) + X_test
# print(f)

# r = []
# for x in range(len(pred_inverse)):
#   r.append(pred_inverse[x+1:x+2] + X_test[x:x+1])

plt.figure(figsize=(12, 5))
plt.xlabel("Date")

ax1 = X_test.Gold.plot(color='blue', grid = True, label = 'Actual')
ax2 = f.Gold.plot(color='red', grid = True, label = 'Predicted', secondary_y = True)

ax1.legend(loc=1)
ax2.legend(loc=2)
plt.title = ('Predicted vs Real')
#plt.show()
plt.savefig("Chapter09/adam_munkaja/Predicted vs Real_1.png")

plt.figure(figsize=(12, 5))
plt.xlabel("Date")

ax1 = X_test.Oil.plot(color='blue', grid = True, label = 'Actual')
ax2 = f.Oil.plot(color='red', grid = True, label = 'Predicted', secondary_y = True)

ax1.legend(loc=1)
ax2.legend(loc=2)
plt.title = ('Predicted vs Real')
#plt.show()
plt.savefig("Chapter09/adam_munkaja/Predicted vs Real_2.png")