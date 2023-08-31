#!/usr/bin/env python
# coding: utf-8

# # Principle Component Analysis (PCA) for Data Visualization
# 

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


# In[3]:


# loading dataset into Pandas DataFrame
df = pd.read_csv(url
                 , names=['sepal length','sepal width','petal length','petal width','target'])


# In[4]:


df.head()


# # Standardize the Data
# 
# Since PCA yields a feature subspace that maximizes the variance along the axes, it makes sense to standardize the data, especially, if it was measured on different scales. Although, all features in the Iris dataset were measured in centimeters, let us continue with the transformation of the data onto unit scale (mean=0 and variance=1), which is a requirement for the optimal performance of many machine learning algorithms.

# In[5]:


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values

y = df.loc[:,['target']].values

x = StandardScaler().fit_transform(x)

pd.DataFrame(data = x, columns = features).head()


# # PCA Projection to 2D
# 

# In[6]:


pca = PCA(n_components=2)


# In[7]:


principalComponents = pca.fit_transform(x)


# In[8]:


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[14]:


principalDf.head(7)


# In[10]:


df[['target']].head()


# In[15]:


finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf.head(7)


# # Visualize 2D Projection
# Use a PCA projection to 2d to visualize the entire data set. You should plot different classes using different colors or shapes. Do the classes seem well-separated from each other?

# In[21]:


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# The three classes appear to be well separated!
# 
# iris-virginica and iris-versicolor could be better separated, but still good!

# # Explained Variance
# The explained variance tells us how much information (variance) can be attributed to each of the principal components.

# In[13]:


pca.explained_variance_ratio_


# Together, the first two principal components contain 95.80% of the information. The first principal component contains 72.77% of the variance and the second principal component contains 23.03% of the variance. The third and fourth principal component contained the rest of the variance of the dataset.
# 
# 

# In[ ]:





# In[ ]:




