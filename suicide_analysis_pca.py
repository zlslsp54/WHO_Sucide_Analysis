#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



#0. Load  suicide pattern data
data= pd.read_csv('suicide_mortrate2_final.csv') 

features = ['pesticide', 'other_poision','hanging','drowning', 'firearms','jumping']


# 1. separating out the features
x = data.loc[:, features].values

# 2. Standardizing the features
#    standardize the dataset’s features onto unit scale (mean = 0 and variance = 1)
x = StandardScaler().fit_transform(x)

# 3. PCA anlaysis
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


#4. Draw PCA plot
finalDf = pd.concat([principalDf, data[['iso_a2']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

ax.scatter(finalDf.loc[:, 'principal component 1']
               , finalDf.loc[:, 'principal component 2']
               , c = 'black'
               , s = 50)

for x,y,text in zip(finalDf.loc[:,'principal component 1'],finalDf.loc[:,'principal component 2'],finalDf.loc[:,'iso_a2'].tolist()):
    ax.text(x+0.1, y, text)

ax.grid()



# 5. Draw PCA plot with group
finalDf = pd.concat([principalDf, data[['whoregion']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['AFR', 'AMR', 'EMR','EUR','SEA','WPR']
colors = ['r', 'y', 'g', 'c','b','pink']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['whoregion'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50
              )
ax.legend(targets)
ax.grid()





