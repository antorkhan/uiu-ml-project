#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
from sklearn import tree
import feature_extractor
from importlib import reload # reload 
reload(feature_extractor)
import glob
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


# In[33]:


def read_files(projects):
    DF = []
#     projects = ['ant']
#     projects = ['ant','camel','ivy','jedit','log4j', 'lucene','poi','synapse','velocity','xalan','xerces']
    for project in projects:
        all_files = glob.glob(f"bug_data/{project}/*.csv")
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            DF.append(df)
    return pd.concat(DF, axis=0, ignore_index=True).drop("name", axis=1)
    
train_dataset = read_files(['ant','camel','ivy','jedit','log4j', 'lucene','poi','synapse','velocity','xalan']).to_numpy()
validation_dataset = read_files(['xerces']).to_numpy()

# X, y = np.split(dataset,[-1],axis=1)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# reg = LogisticRegression(max_iter=5000).fit(x_train, y_train.reshape(y_train.shape[0],))
# reg.score(x_test, y_test.reshape(y_test.shape[0],))


# In[34]:


ht = HoeffdingTreeClassifier(no_preprune=True)
X_train, y_train = np.split(train_dataset,[-1],axis=1)
X_train = scaler.fit_transform(X_train)
y_train = np.where(y_train.reshape(y_train.shape[0]) > 0, 1, 0)

X_validation, y_validation = np.split(validation_dataset,[-1],axis=1)
X_validation = scaler.fit_transform(X_validation)
y_validation = np.where(y_validation.reshape(y_validation.shape[0]) > 0, 1, 0)

remaining = X_train.shape[0]
starting = 0
CHUNK = 200
while(remaining != 0):
    chunk_size = min(remaining, CHUNK)
    remaining -= chunk_size
    starting += chunk_size
    ht = ht.partial_fit(X_train[starting:starting+chunk_size], y_train[starting:starting+chunk_size])
    y_prediction = ht.predict(X_validation)
    print(accuracy_score(y_validation, y_prediction))


# In[35]:


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

X_train, y_train = np.split(train_dataset,[-1],axis=1)
X_train = scaler.fit_transform(X_train)
y_train = np.where(y_train.reshape(y_train.shape[0]) > 0, 1, 0)

X_validation, y_validation = np.split(validation_dataset,[-1],axis=1)
X_validation = scaler.fit_transform(X_validation)
y_validation = np.where(y_validation.reshape(y_validation.shape[0]) > 0, 1, 0)

remaining = X_train.shape[0]
starting = 0
CHUNK = 200
while(remaining != 0):
    chunk_size = min(remaining, CHUNK)
    remaining -= chunk_size
    starting += chunk_size
    ht = clf.fit(X_train[starting:starting+chunk_size], y_train[starting:starting+chunk_size])
    y_prediction = ht.predict(X_validation)
    print(accuracy_score(y_validation, y_prediction))
