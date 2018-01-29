#hc

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

#using the dendogram to find the optimal no of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward')) #ward method tries to decrease the variance in our clusters
plt.title('dendogram')
plt.xlabel('no of clusters')
plt.ylabel('euc dist')
plt.show()

#fitting hc to dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(X)

#visualising the results
#same as in kmeans'