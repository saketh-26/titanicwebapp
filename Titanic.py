#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import the packages 
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings("ignore")


# # Loading the dataset

# In[3]:


data = pd.read_csv("titanic.csv")
#data.head()


# # Performing EDA 

# In[4]:


#data.columns


# In[5]:


data.isnull().sum() #checking the count of missing values


# In[6]:


#First let's understand important columns information
data['Survived'].value_counts() #gives the count of unique values


# In[8]:


#same for rmng columns
data['Pclass'].value_counts()


# In[9]:


data['Sex'].value_counts()


# In[10]:


data['Embarked'].value_counts()


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns
#we will get the unique value count
data['Survived'].value_counts().plot(kind="bar")
#plt.show()


# In[47]:


data['Embarked'].value_counts().plot(kind='pie')
#plt.show()


# In[48]:


sns.boxplot(x="Embarked",y="Age",data=data) #box plot -->Quartile Distribution
#plt.show()


# In[49]:


sns.boxplot(x="Sex",y="Age",data=data)
#plt.show()


# In[50]:


#we can get the count also on histograms -->bar_label
counts, edges, bars = plt.hist(data['Age'],color='blue')
#plt.bar_label(bars)
#plt.show()


# In[18]:


#Filling the Age column -->as majority age group is from 20-30 we fill with mean
data['Age'].fillna(data['Age'].mean(),inplace=True) #inplace=True will make permanent changes


# In[19]:


data.drop(columns=['Cabin'],inplace=True)


# In[20]:


#data.info()


# In[21]:


#We are having 2 missing values in embarked we can get it
data.loc[data['Embarked'].isnull()]


# In[22]:


data['Embarked'].fillna('S',inplace=True)


# In[23]:


#Feature Encoding -->converting categorical data to numerical values
sex = pd.get_dummies(data['Sex'],drop_first=True)
#sex
#do for Pclass,Embarked


# In[24]:


pclass = pd.get_dummies(data['Pclass'],drop_first=True)
#pclass


# In[25]:


embarked = pd.get_dummies(data['Embarked'],drop_first=True)
#embarked


# In[26]:


#Combining all the dataframes
final_data = pd.concat([data,sex,pclass,embarked],axis='columns')
#final_data


# In[27]:


#FInally we will drop unnecessary columns
final_data.drop(columns=['PassengerId','Sex','Name',
                         'Pclass',
                         'Ticket',
                         'Embarked'],inplace=True)


# In[28]:


#final_data


# # Building the Model

# In[31]:


#Training and Testing data
X = final_data.drop('Survived',axis=1)
y = final_data["Survived"]


# In[32]:


X.columns


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


#Splitting the data into training and testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,
                                                 random_state=1) 


# In[36]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[37]:


#predictors
predictions = logmodel.predict(X_test)


# In[51]:


#Final Metrics Calculation
from sklearn.metrics import plot_confusion_matrix,accuracy_score,confusion_matrix,classification_report,roc_curve
#print(accuracy_score(y_test,predictions)*100)


# In[52]:


#print(confusion_matrix(y_test,predictions))


# In[53]:


#print(classification_report(y_test,predictions))


# In[55]:


#plot_confusion_matrix(model,x_train,y_train,values_format='d')
#plot_confusion_matrix(logmodel,X_test,y_test,values_format='d')
#plt.show()


# In[45]:


#prediction on test data
#logmodel.predict([[35.000000,1,0,53.1000,0,0,0,0,1]])


def survive(arr):
    predictions = logmodel.predict(arr)
    return predictions




