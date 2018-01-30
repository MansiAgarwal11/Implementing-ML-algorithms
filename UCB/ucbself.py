#ucb
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
import math
d=10 #np of ads
N= 10000 #no of rounds
ads_selected=[] #index of ad selected in each round 
number_of_selections= [0]*d #storing no of times each ad was selected in each round
sum_of_rewards=[0]*d #storing the no of times each ad was clicked by the user in each round
total_reward=0
for n in range(0,N):
    max_upper_bound=0
    ad=0
    for i in range(0,d):
        if(number_of_selections[i]>0):
            average_reward=sum_of_rewards[i]/number_of_selections[i] 
            delta_i= math.sqrt(3/2 * math.log(i+1) / number_of_selections[i] )
            upper_bound= delta_i + average_reward
        else:
            upper_bound=1e400
        if max_upper_bound< upper_bound:
            max_upper_bound=upper_bound
            ad=i
    ads_selected.append(ad)
    number_of_selections[ad]=number_of_selections[ad] +1
    reward=dataset.values[n,ad]
    sum_of_rewards[ad]=sum_of_rewards[ad]+ reward
    total_reward=total_reward+reward
    
#plotting results
plt.hist(ads_selected)
        
        
