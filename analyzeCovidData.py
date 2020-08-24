#analyses the data using the ARIMA model

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import sys
import os
import glob
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from pandas import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


#code taken from https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order) #fit ARIMA model using current history (current training set)
		model_fit = model.fit(disp=0) #disp=0 prevents fitting output from showing
		yhat = model_fit.forecast()[0] #get first value of forecast into the future->one of the values corresponding to test set
		predictions.append(yhat) #add this predicted yhat value to the predictions
		history.append(test[t]) #add this predicted yhat value to the history, we have more information for our ARIMA model
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
	return (best_cfg, best_score)

def num(s):
	try:
		val=float(s) #ARIMA only takes float values
		return True
	except:
		return False

def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')

##internationalDeathsDates=[]
##internationalDeathsValuesAtEachDate=None
##
##laDeathsDates=[]
##laDeathsValuesAtEachDate=None
##
##
##countiesDeathsDates=[]
##countyDeathsValuesAtEachDate=None

def processData(file):
        with open(file) as f:
                dates=[]
                valuesAtEachDate=[]
                i=0
                for line in f:
                        if i==0:
                                potentialDates=line.strip().split(',')
                                if (potentialDates[0]==""):
                                        dates=potentialDates[1:]
                                dates=[parser(x) for x in dates]
                                numberOfDates=len(dates)
                                valuesAtEachDate=[0.0] * numberOfDates
                                print(len(dates))
                                i+=1
                        else:
                                numberOfDeathsInASubareaPerDate=line.strip().split(',')
                                index=0
                                for i in range(len(numberOfDeathsInASubareaPerDate)):
                                        if (num(numberOfDeathsInASubareaPerDate[i])):
                                                valuesAtEachDate[index]+=float(numberOfDeathsInASubareaPerDate[i])
                                                index+=1
                                #print(index)
                valuesAtEachDate=np.array(valuesAtEachDate)
                deathSeries=pd.Series(valuesAtEachDate, index=dates)
                return deathSeries

##internationalDeathsValuesAtEachDate=np.array(internationalDeathsValuesAtEachDate)
##internationalDeathsSeries=pd.Series(internationalDeathsValuesAtEachDate, index=internationalDeathsDates)
##print(internationalDeathsSeries)
internationalDeathsSeries=processData("Data/International/International_covid_deaths_data.csv")
countyDeathsSeries=processData("Data/US Counties/US_county_covid_deaths_data.csv")
#laCasesSeries=processData("Data/LA/LA_cities_covid_data.csv")
#internationalCasesSeries=processData("Data/International/International_covid_cases_data.csv")
#countyCasesSeries=processData("Data/US Counties/US_county_covid_cases_data.csv")

with open("internationalDeaths.csv","w") as file:
        internationalDeathsSeries.to_csv(file)

with open("countyDeathsSeries.csv", "w") as file:
        countyDeathsSeries.to_csv(file)
        


##countyDeathsSeries.plot()
##plt.show()
##autocorrelation_plot(countyDeathsSeries)
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()
##plot_pacf(countyDeathsSeries, lags=50)
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()
##diffInt = internationalDeathsSeries.diff()
##diffCounty=countyDeathsSeries.diff()
##plt.plot(diffCounty)
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()

        
##plt.plot(diffInt)
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()
##
##diffInt.plot()
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()
##autocorrelation_plot(diffInt)
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()
##plot_pacf(diffInt, lags=50)
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()

##
##p=range(5)
##q=range(5)
##d=range(1,2,1)
##evaluate_models(countyDeathsSeries.values, p, d, q)

##internationalDifferenceNumber=1
##internationalDeathsQ=50 #without any differencing
##internationalDeathsP=1
##internationalDeathsD=1


#intDecomposition=seasonal_decompose(internationalDeathsSeries)
model = ARIMA(internationalDeathsSeries, order=(4,1,1))
resultsInt = model.fit(disp=0)
resultsInt.plot_predict(1, 177+300)
plt.show()
plt.clf()
plt.close()


##model = ARIMA(diffInt, order=(4,1,1))
##resultsInt = model.fit(disp=0)
##resultsInt.plot_predict(1, 177+300)
##plt.show()
##plt.clf()
##plt.close()
##
##intLog=np.log(internationalDeathsSeries)

model = ARIMA(countyDeathsSeries, order=(4,1,2))
resultsInt = model.fit(disp=0)
resultsInt.plot_predict(1, 175+300)
plt.show()
plt.clf()
plt.close()


#sample code for viewing autocorrelation, partial autocorrelation, original plot, and differentcing

####fig = plt.figure()
####ax = plt.subplot(111)
####ax.xaxis.set_major_locator(plt.MaxNLocator(6))
####ax.plot(internationalDeathsSeries)
####fig.savefig("internationalDeathsSeries.png")
####
##plt.cla()
##plt.clf()
##plt.close()
##
##
##internationalDeathsSeries.plot()
##plt.show()
##autocorrelation_plot(internationalDeathsSeries)
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()
##plot_pacf(internationalDeathsSeries, lags=50)
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()
##diff = internationalDeathsSeries.diff()
##plt.plot(diff)
##plt.show()
##plt.cla()
##plt.clf()
##plt.close()
####
