# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder() #LABELENCODER HELPS TO CATEGORISE DATA INTO 0,1,2 ..
X[:,3] =labelencoder_X.fit_transform(X[:,3]) 
onehotencoder=OneHotEncoder(categorical_features = [3]) #ONEHOTENCODER TRANFORMS THE CATEGORIACL DATA INTO VECTORS OF O AMD 1S
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
X= X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#We dont need to do feature scaling. our multiple linear library takes care of it
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting multiple linear regressor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#testing

#building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#we are doing the following as the backward elimination doesnt take the constant theto not in use and therefor
#we introduce another column x not =1
X= np.append(arr= np.ones((50,1)), values=X , axis = 1)
X_opt= X[:, [0,1,2,3,4,5]]
regressor_ols= sm.OLS(endog =y, exog= X_opt ).fit()
regressor_ols.summary()
X_opt= X[:, [0,1,3,4,5]]
regressor_ols= sm.OLS(endog =y, exog= X_opt ).fit()
regressor_ols.summary()
X_opt= X[:, [0,3,4,5]]
regressor_ols= sm.OLS(endog =y, exog= X_opt ).fit()
regressor_ols.summary()
X_opt= X[:, [0,3,5]]
regressor_ols= sm.OLS(endog =y, exog= X_opt ).fit()
regressor_ols.summary()
X_opt= X[:, [0,3]]
regressor_ols= sm.OLS(endog =y, exog= X_opt ).fit()
regressor_ols.summary()
