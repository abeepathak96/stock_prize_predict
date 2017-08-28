

##### ====== IMPORTS =====#####
import csv
import pandas as pd
import math
from datetime import datetime as ds
from datetime import datetime, date, time, timedelta
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numba.tests.test_array_constants import dt
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates

#-- Using The Matplotlib Fonts --#

style.use('ggplot')

file = open('TSLA.csv', "rb") # -- this the csv file input that contains the stock prizes of previouse years --- #
df = pd.read_csv(f, delimiter = ",") # --- this is the read function invoked from the pandas library to read the csv file and here we are adding the csv file innto the pandas dataframe. Dataframe is a row and column datatype in pandas in python---#

df = df[['Date','Open','High','Low','Close','Volume']] #-- here we are adding specific columns in the pandas dataframe. We rae adding Date, Open, High, Low, CLose, Volume columns as this will used to form features ----#

df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0 # --- Here we are creating another colunmn from existing coulmn as a attribue this coulmn is a High-Low percentage --#
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0 #---- Here we are creating another column as a attribute using Percentage change between Close and Open --#

df = df[['Date', 'Close', 'HL_PCT', 'PCT_change','Volume']] # ---- here we are gain configuring dataframe --#
forecast_col = 'Close'
df.fillna(value=-99999, inplace=True)
df.to_csv('preprocessed_data.csv')
df.plot()
plt.show()

df['Close'].plot()
plt.show()

forecast_out = int(math.ceil(0.1 * len(df)))
df['label'] = df[forecast_col].shift(periods=forecast_out)
        #print(df.head())

X = np.array(df.drop(['label', 'Date'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
        #clf = svm.SVR(kernel='linear')
        #clf.fit(X_train, y_train)
        #confidence = clf.score(X_test, y_test)
        #print(confidence)
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

df.to_csv('tested_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
last_recent_date = df['Date'].min()
last_date = df['Date'].max()

last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = ds.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]


ax = df[['Close', 'Forecast']].plot(kind='bar', title="status Bar", figsize=(15, 10), stacked=True,legend=True, fontsize=1)
ax.set_xlabel("Date", fontsize=2)
ax.set_ylabel("Price", fontsize=12)
plt.tight_layout()
plt.show()
df = df[['Date', 'Forecast']]
df.plot()
plt.show()
df.to_csv('output.csv')

