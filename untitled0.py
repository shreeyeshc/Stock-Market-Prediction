import csv                              #importing csv file
import numpy as np                      #importing array in np
from sklearn.svm import SVR             #Support vector regression
import matplotlib.pyplot as plt         #ploting in figure
import time                             #import time

dates = []                              
prices = []

def get_data(filename):
    with open(filename, 'r') as csvFile:           #opening csv file
        csvFileReader = csv.reader(csvFile)           #reading cvs file
        start_time = time.time()
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))          #format of date(06-Jan-16)
            prices.append(float(row[1]))
        print("---- COMPLETED READING FILE: %s IN %s SECONDS ---"  % (filename,time.time() - start_time))
        print(dates)
        return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates),1))
    svr_lin = SVR(kernel = 'linear', C=1e3)            #linear regression
    svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma = 0.1)          #radial basis function
    
    start_time = time.time()
    svr_lin.fit(dates,prices)                  #ploting  linear into the graph
    print("--- LINEAR FIT COMPLETE IN %s SECONDS ---" % (time.time() - start_time))
    print("\n")
        
    start_time = time.time()
    svr_rbf.fit(dates,prices)              #ploting rbf into the graph
    print("--- RBF FIT COMPLETE IN %s SECONDS ---" % (time.time() - start_time))
    print("\n")
    
    rbf_prediction = svr_rbf.predict(x)[0],            #computing rbf
    linear_prediction = svr_lin.predict(x)[0]          #computing linear regression
    
    print("RBF PREDICTION IS: ",rbf_prediction)            #printing rbf value
    print("\n")
    print("LINEAR PREDICTION IS: ",linear_prediction)           #printing linear regression value
    plt.scatter(dates,prices,color='black', label='Data')
    plt.plot(dates,svr_rbf.predict(dates), color = 'red', label = 'RBF model')
    plt.plot(dates,svr_lin.predict(dates), color = 'blue', label = 'Linear model')
    plt.xlabel('Dates')                    #labels
    plt.ylabel('Price')
    plt.title('Support Vector Regression')         #title of graph
    plt.legend()
    plt.show()     
    return rbf_prediction, linear_prediction

get_data('eicher.csv')           #cvs file name

predicted_price = predict_prices(dates,prices,31)       #predicted value


