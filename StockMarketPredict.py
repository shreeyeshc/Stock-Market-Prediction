import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import time

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvFile:
		csvFileReader = csv.reader(csvFile)
		start_time = time.time()
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	print("---- COMPLETED READING FILE: %s IN %s SECONDS ---" % (filename,time.time() - start_time))
	print(dates)
	return

def predict_prices(dates, prices, x):
	dates = np.reshape(dates,(len(dates),1))
	svr_lin = SVR(kernel = 'linear', C=1e3)

	start_time = time.time()
	svr_lin.fit(dates,prices)
	print("--- LINEAR FIT COMPLETE IN %s SECONDS ---" % (time.time() - start_time))
	print("\n")
    

	plt.scatter(dates,prices,color='black', label='Data')
	plt.plot(dates,svr_lin.predict(dates), color = 'blue', label = 'Linear model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Linear Regression')
	plt.legend()
	plt.show()
    
	linear_prediction = svr_lin.predict(x)[0]


	print("STOCK PREDICTION IS: ",linear_prediction)
	return linear_prediction


get_data('mrff.csv')

predicted_price = predict_prices(dates,prices,31)


