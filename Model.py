#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[20]:


df = pd.read_csv('C:/Users/rahul03/SpyderProjects/Housing_Price_Predictor/Housing_Data.csv')
df.head(5)


# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[38]:


columns = df.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Price"]]
# Store the variable we are predicting 
target = "Price"
# Define a random state 
state = np.random.RandomState(42)
X = df[columns]
Y = df[target]


# In[40]:


X.head(5), Y.head(5)


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, shuffle = True, random_state = 0)


# In[44]:


y_train


# In[45]:


from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)


# In[46]:


print ("R^2 is: \n", model.score(X_test, y_test))


# In[47]:


predictions = model.predict(X_test)


# In[48]:


actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')


# In[49]:



X_test


# In[53]:


model.predict([[5, 1.5]])

