# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 23:57:41 2017

@author: Mansi Agarwal
"""


#importing the libraries
import numpy as np  #CONTAINS MATHS TOOLS
import matplotlib.pyplot as plt #PLOTTING CHARTS
import pandas as pd #IMPORTING AND MANAGING DATA

#importing the dataset
dataset= pd.read_csv('Data.csv') 
X=dataset.iloc[:,:-1].values #INDEPENDENT VARIABLES
Y=dataset.iloc[:,3].values   #DEPENDENT VARIABLES

#Handling missing data
from sklearn.preprocessing import Imputer #IMPUTER HANDLES MISSING DATA
imputer= Imputer(missing_values= 'NaN', strategy= 'mean' ,axis =0)
#TODO IMPUTER=IMPUTER.. WHY?
imputer=imputer.fit(X[:,1:3]) #upper bound is not included, this means col 1 and 2 
X[:,1:3]=imputer.transform(X[:,1:3])

#use np.set_printoptions(threshold=np.nan) to view the full array

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder() #LABELENCODER HELPS TO CATEGORISE DATA INTO 0,1,2 ..
X[:,0] =labelencoder_X.fit_transform(X[:,0]) 
onehotencoder=OneHotEncoder(categorical_features = [0]) #ONEHOTENCODER TRANFORMS THE CATEGORIACL DATA INTO VECTORS OF O AMD 1S
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y= LabelEncoder()
Y= labelencoder_Y.fit_transform(Y) #Y HERE IS A VECTOR AND HENCE NO [:,0] REQD


#Splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y , test_size= 0.2, random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test=  sc_X.transform(X_test)  #TODO DONT NEED TO FIT HERE BUT WHY?
#AS THIS IS A CLASIFICATION PROBLEM, WE DONT NEED TO FEATURE SCALE Y BUT IN LINEAR REGRESSION WE WILL

