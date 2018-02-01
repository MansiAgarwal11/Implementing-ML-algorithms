#Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
import random

d=10 #np of ads
N= 10000 #no of rounds
ads_selected=[] #index of ad selected in each round 
number_of_rewards_1= [0]*d #storing no of times each ad was clicked in each round
number_of_rewards_0=[0]*d #storing the no of times each ad wasnot clicked in each round
total_reward=0
for n in range(0,N):
    max_random=0
    ad=0
    for i in range(0,d):
        random_beta=random.betavariate(number_of_rewards_1[i] +1, number_of_rewards_0[i] +1) #gives us random draws of the beta distributn of the parameters that we choose
        if max_random< random_beta:
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if reward==1:
        number_of_rewards_1[ad]=number_of_rewards_1[ad]+1
    else:
        number_of_rewards_0[ad]=number_of_rewards_0[ad]+1
    total_reward=total_reward+reward
    
#plotting results
plt.hist(ads_selected)
        
        
