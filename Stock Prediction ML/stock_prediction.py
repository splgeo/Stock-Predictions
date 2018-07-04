# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:31:38 2018

@author: david
"""
import datetime
from datetime import date
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
import pandas as pd
from sklearn.svm import SVR
import csv
import numpy as np

stock = input('Enter stock symbol')
start = datetime.date.today() + datetime.timedelta(-30)
end = str(date.today())

# get stock data 
data = yf.download(stock, start, end)
data.to_csv('stock.csv')
data = pd.read_csv('stock.csv')
cols = ['Close']
df = data[cols]
df.to_csv('stock.csv')

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	
		for row in csvFileReader:
			dates.append(int(row[0]))
			prices.append(float(row[1]))
	return

def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) 
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) 
	svr_rbf.fit(dates, prices) 
	plt.plot(dates, prices, color= 'black', label= 'Data')  
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') 
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()
	return svr_rbf.predict(x)[0]

get_data('stock.csv') 


predicted_price = predict_price(dates, prices, 30) 
print(predicted_price)