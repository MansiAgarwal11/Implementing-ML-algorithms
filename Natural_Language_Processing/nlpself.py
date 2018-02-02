#NLP

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3) #delimiter is tab and quoting=3 is used to avoid doublequotes

#Cleaning the texts
import re
import nltk #nltk contains the list of useless words known as stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,1000):
    review= re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) #review is thenew version of the original review  (the cleaned version)
    #re.sub('[^a-zA-Z]') will remove the numbers and the punctuation marks, only alphabets exist and in place of the removed texts a space will be provided 

    review=review.lower()
    #this will convert all the uppercase alpha to lowercase
    
    #removing all the useless words that dont determine whether a review is +ve or -ve
    review=review.split() #review becomes a list of diff items(words)
    #stemming is taking the root of the word
    review= [ps.stem(word) for word in review if not word in set( stopwords.words('english'))]
    #set is used to make the process faster as in lists 

    #converting list of words back into string
    review=' '.join(review)
    corpus.append(review)
    
#Creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray() #dependent variable matrix(sparse)
y=dataset.iloc[:,1].values  #independent variable vector

#classification model-naive bayes
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

# Fitting random forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10, criterion='entropy' , random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
