#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/bibif/Downloads/winequality-red.csv")
df.head()


# In[3]:


correlations = df.corr()['quality'].drop('quality')
print(correlations)


# In[4]:


sns.heatmap(df.corr())
plt.show()


# In[5]:


def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations


# In[6]:


features = get_features(0.05) 
print(features) 
x = df[features] 
y = df['quality']


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)


# In[8]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_)


# In[9]:


train_pred = regressor.predict(x_train)
print(train_pred)
test_pred = regressor.predict(x_test) 
print(test_pred)


# In[10]:


predicted_data = np.round_(test_pred)
print(predicted_data)


# In[11]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))


# In[12]:


print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))


# In[13]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))


# In[14]:


# displaying coefficients of each feature
coeffecients = pd.DataFrame(regressor.coef_,features) 
coeffecients.columns = ['Coeffecient'] 
print(coeffecients)

