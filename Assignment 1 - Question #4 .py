#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[16]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
print("Setup Complete")


# # Read data

# ## Specify path of dataset

# In[17]:


my_path = './hcvdat0.csv'


# In[18]:


hcv_data = pd.read_csv(my_path)
hcv_data.head()


# ## Make sure class column is the last column in data-frame

# In[19]:


cols = hcv_data.columns.tolist()
cols.remove('Category')
cols.append('Category')
hcv_data = hcv_data[cols]
hcv_data


# # Scatter Plots

# ## ALP-PROT
# We can see most of our data members have '0=Blood Donor' class. Also members of '2=Fibrosis' and '1=Hepatitis' class, mostly have 'ALP' less than 50

# In[20]:


plt.figure(figsize=(10,7))
sns.scatterplot(x=hcv_data['ALP'], y=hcv_data['PROT'], hue=hcv_data['Category'])
plt.show()


# ## ALP-CREA
# We can see, classes are distributed evenly, but ages between 30-60, hold hold most of blood donor counts.

# In[21]:


plt.figure(figsize=(10,7))
sns.scatterplot(x=hcv_data['GGT'], y=hcv_data['Age'], hue=hcv_data['Category'])
plt.show()


# ## ALT-PROT
# Data members of class '3=Cirrhosis' have less 'ALT' values than others.

# In[22]:


plt.figure(figsize=(10,7))
sns.scatterplot(x=hcv_data['ALT'], y=hcv_data['PROT'], hue=hcv_data['Category'])
plt.show()


# # Decsion Tree

# ## 10-fold validation
# Since last fold, has less score than others, could suspect that are model has been over-fitted by our data. Another cause of first and last fold scores, being less than others, might be outliers. 

# In[23]:


# Drop rows containing NAN
hcv_data = hcv_data.dropna(axis = 0)
x = hcv_data[hcv_data.columns[:-1]]
y = hcv_data['Category']
# Remove 'Sex', because it's categorical 
x = x.drop('Sex', axis=1)
clf = tree.DecisionTreeClassifier()
# Perform 10-fold cross validation 
scores = cross_val_score(estimator=clf, X=x, y=y, cv=10, n_jobs=4)
print(scores)


# # Preprocessing

# ## Find categorical variables

# In[24]:


# Get list of categorical variables
s = (hcv_data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# ## Label encode categorical variables

# In[25]:


from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
hcv_numerical = hcv_data.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    hcv_numerical[col] = label_encoder.fit_transform(hcv_data[col])
hcv_numerical


# ## Normalize data columns

# In[26]:


from sklearn import preprocessing

x_data = hcv_numerical.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x_data)
df = pd.DataFrame(x_scaled)
df.columns = hcv_numerical.columns
df.head()


# # KNN classifier
# We can see that scores of CV has improved for KNN classifier. scores are almost the same with an acceptable mean value for them.

# In[27]:


from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, hcv_numerical[hcv_numerical.columns[:-1]], hcv_numerical['Category'], cv=10)
#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

