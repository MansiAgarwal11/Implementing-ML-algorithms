#simple linear regression

#importing the libraries
import numpy as np  #CONTAINS MATHS TOOLS
import matplotlib.pyplot as plt #PLOTTING CHARTS
import pandas as pd #IMPORTING AND MANAGING DATA

#importing the dataset
dataset= pd.read_csv('Salary_Data.csv') 
X=dataset.iloc[:,:-1].values #INDEPENDENT VARIABLES
Y=dataset.iloc[:,1].values   #DEPENDENT VARIABLES

#Splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y , test_size= 1/3, random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test=  sc_X.transform(X_test) """

#Fitting simple kinear regression to the training data set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, Y_train) #regressor is our ml model


#predicting the test set results
Y_pred=regressor.predict(X_test)

#visualising the Training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary Vs Experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')

#visualising the Test set results
plt.scatter(X_test,Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue') #no change here as we have already predicted our hypothsis and dont wnat to change it
plt.title('Salary Vs Experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
