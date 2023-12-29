#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[4]:


dataset = pd.read_csv('HousingData.csv')
dataset.head()


# In[15]:


dataset.info()


# In[10]:


dataset.describe()


# In[11]:


dataset.isnull().sum()


# In[70]:


dataset['CRIM'].fillna(int(dataset['CRIM'].mean()),inplace=True)
dataset['ZN'].fillna(int(dataset['ZN'].mean()),inplace=True)
dataset['INDUS'].fillna(int(dataset['INDUS'].mean()),inplace=True)
dataset['CHAS'].fillna(int(dataset['CHAS'].mean()),inplace=True)
dataset['AGE'].fillna(int(dataset['AGE'].mean()),inplace=True)
dataset['LSTAT'].fillna(int(dataset['LSTAT'].mean()),inplace=True)


# In[71]:


dataset.describe()


# In[72]:


fig,ax = plt.subplots(ncols=7,nrows=2,figsize=(20,10))
index = 0
ax = ax.flatten()
for col,value in dataset.items():
    sns.boxplot(y=col,data=dataset,ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5,w_pad=0.7,h_pad=0.5)


# In[73]:


fig,ax = plt.subplots(ncols=7,nrows=2,figsize=(20,10))
index = 0
ax = ax.flatten()
for col,value in dataset.items():
    sns.displot(value,ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5,w_pad=0.7,h_pad=0.5)


# In[74]:


#min-max Normalization
cols = ['CRIM','ZN','TAX','B']
for col in cols:
    #find min and max of that column
    minimum = min(dataset[col])
    maximum = max(dataset[col])
    dataset[col] = (dataset[col]-minimum)/(maximum - minimum)


# In[79]:


fig,ax = plt.subplots(ncols=7,nrows=2,figsize=(20,10))
index = 0
ax = ax.flatten()
for col,value in dataset.items():
    sns.displot(value,ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5,w_pad=0.7,h_pad=0.5)


# In[78]:


#standardization
from sklearn import preprocessing
scalar = preprocessing.StandardScaler()

scaled_cols = scalar.fit_transform(dataset[cols])
scaled_cols = pd.DataFrame(scaled_cols,columns=cols)
scaled_cols.head()


# In[77]:


for col in cols:
    dataset[col] = scaled_cols[col]


# In[76]:


fig,ax = plt.subplots(ncols=7,nrows=2,figsize=(20,10))
index = 0
ax = ax.flatten()
for col,value in dataset.items():
    sns.displot(value,ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5,w_pad=0.7,h_pad=0.5)


# In[80]:


#coorelation matrix
cor = dataset.corr()
plt.figure(figsize=(25,15))
sns.heatmap(cor, annot=True,cmap='coolwarm')


# In[81]:


sns.regplot(y=dataset['MEDV'],x=dataset['LSTAT'])


# In[82]:


sns.regplot(y=dataset['MEDV'],x=dataset['RM'])


# In[83]:


dataset['RAD'].isnull().sum()


# In[84]:


#input split
x = dataset.drop(columns=['MEDV','RAD'],axis=1)
y = dataset['MEDV']


# In[85]:


x.isnull().sum()


# In[99]:


#model training
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
def train(model,x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42)
    model.fit(x,y)
    
    #predict the training set
    pred = model.predict(x)
    
    #perform cross validation
    cv_score = cross_val_score(model,x,y,scoring='neg_mean_squared_error',cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("model Report")
    print("MSE",mean_squared_error(y,pred))
    print('cv_score',cv_score)


# In[100]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
train(model,x,y)
coef = pd.Series(model.coef_, x.columns).sort_values()
coef.plot(kind='bar',title='model coefficients')


# In[103]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model,x,y)
coef = pd.Series(model.feature_importances_,x.columns).sort_values(ascending=False)
coef.plot(kind='bar',title='Feature Importance')


# In[105]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
train(model,x,y)
coef = pd.Series(model.feature_importances_,x.columns).sort_values(ascending=False)
coef.plot(kind='bar',title='Feature Importance')


# In[107]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
train(model,x,y)
coef = pd.Series(model.feature_importances_,x.columns).sort_values(ascending=False)
coef.plot(kind='bar',title='Feature Importance')

