#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import glob
from sklearn.model_selection import train_test_split
#Accuracy score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import tree


# In[82]:


def update_current_data_window(current_project_data):
    global CURRENT_DATA_WINDOW, WINDOW_SIZE
    if(CURRENT_DATA_WINDOW is None):
        CURRENT_DATA_WINDOW = pd.DataFrame(current_project_data)
        return
        
#     Keep last 1000 historical data
#     print(f"before: {current_data_window}")
    CURRENT_DATA_WINDOW = CURRENT_DATA_WINDOW.append(current_project_data).tail(WINDOW_SIZE)
#     print(f"updated data window, current shape is: {current_data_window.shape}")
#     print(f"after: {current_data_window}")
    return
def get_data(start, count, project_data):
    return project_data[start: start+count]

def split_X_Y(data_frame):
    return np.split(data_frame,[-1],axis=1)


def update_model():
    global CURRENT_DATA_WINDOW, MODEL
    X, Y = split_X_Y(CURRENT_DATA_WINDOW)
    MODEL.fit(X,Y)

def read_project_data(project):
    DF = []
    all_versions = glob.glob(f"bug_data/{project}/*.csv")
    for version in all_versions:
        df = pd.read_csv(version, index_col=None, header=0)
        DF.append(df)
    data = pd.concat(DF, axis=0, ignore_index=True).drop("name", axis=1)
    data.iloc[:,-1:] = data.iloc[:,-1:].astype(bool).astype(int)
    return data
def get_color(index):
    colors = ['orange','blue','red','magenta','teal', 
              'brown','black','chocolate','chartreuse','deeppink']
    return colors[index]

def get_acc_from_cm(cm, attr='accuracy'):
    tn, fp, fn, tp = cm
    if(attr=='accuracy'):
        return (tp + tn) / (tp + tn + fp + fn)
    elif(attr=='precision'):
        return tp / (tp + fp)


# In[87]:


PROJECTS = ['ant','camel','ivy','jedit','log4j', 'lucene','poi','synapse','velocity','xalan']
MAX_DEPTH = 3
MODEL = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='gini')
CURRENT_DATA_WINDOW = data = None
WINDOW_SIZE = 1500
CHUNK_SIZE = chunk_size = 350
ACCURACY_HISTORY = {}
CM_HISTORY = {}
F1_HISTORY = {}
ITERATION = 0
AGGRETAGED_AVG = 0


for project in PROJECTS:
    data = read_project_data(project)
    starting = 0
    project_iteration = 0
    project_aggregated_avg = 0
    total_remaining = total = data.shape[0]
    ACCURACY_HISTORY[project] = []
    CM_HISTORY[project] = []
    F1_HISTORY[project] = []

#     data[starting: starting+chunk_size]
#     update_model_from_range(starting, chunk_size)
    if(CURRENT_DATA_WINDOW is None): 
        print("Initializing....\n")
        update_current_data_window(get_data(starting, CHUNK_SIZE, data))
        update_model()
        starting += chunk_size
        total_remaining -= CHUNK_SIZE
#         Create model with first n data rows of first project

    while(total_remaining > 0):
        ITERATION += 1
        project_iteration += 1
        current_chunk_size = min(total_remaining, chunk_size)
        cur_data = get_data(starting, current_chunk_size, data)
        cur_x, cur_y = split_X_Y(cur_data)
#         print(cur_data)
        cur_y_pred = MODEL.predict(cur_x)
        update_current_data_window(cur_data)
        starting += current_chunk_size   
        update_model()
        print(f"Project: {project} Batch: {project_iteration} ACC: ",round(accuracy_score(cur_y_pred, cur_y), 4))
       
        ACCURACY_HISTORY[project].append(accuracy_score(cur_y_pred, cur_y))
        CM_HISTORY[project].append(confusion_matrix(cur_y, cur_y_pred).ravel())
        F1_HISTORY[project].append(f1_score(cur_y, cur_y_pred))
        total_remaining -= current_chunk_size
#         print("Moving Average with evolving model: ", AGGRETAGED_AVG/ITERATION)
    print("Project Average Acc with evolving model: ", project_aggregated_avg/project_iteration)
    


# In[88]:


# x_from = 0
# last_x = None
# fig, ax = plt.subplots(figsize=(16, 6))
# ax.set_xlim(xmax=50)

# for index, project in enumerate(PROJECTS):
#     x = list(range(x_from, x_from + len(ACCURACY_HISTORY[project]))) 
#     y = ACCURACY_HISTORY[project]
#     print(x)
#     print(y)
#     if(last_x):
#         ax.plot([last_x,x[0]],[last_y,y[0]],color=get_color(index))
#     last_x = x[-1]
#     last_y = y[-1]
#     ax.plot(x,y,color=get_color(index), label=project)
#     x_from += len(ACCURACY_HISTORY[project])
# plt.legend(loc="lower right", prop={'size': 13})
# plt.ylabel('Accuracy',fontsize=18)
# plt.xlabel('Iteration',fontsize=18)
# # plt.show()
# plt.savefig('project_based_accuracy_evolve.png', dpi=300)


# In[89]:


x_from = 0
last_x = None
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(xmax=50)
attr = 'accuracy'
total = 0
for index, project in enumerate(PROJECTS):
    
    x = list(range(x_from, x_from + len(CM_HISTORY[project]))) 
    y = list(map(lambda cm: get_acc_from_cm(cm,attr), CM_HISTORY[project]))
    total += sum(y)
    if(last_x):
        ax.plot([last_x,x[0]],[last_y,y[0]],color=get_color(index))
    last_x = x[-1]
    last_y = y[-1]
    ax.plot(x,y,color=get_color(index), label=project)
    x_from += len(ACCURACY_HISTORY[project])
plt.legend(loc="lower right", prop={'size': 13})
plt.ylabel(attr, fontsize=18)
plt.xlabel('iteration',fontsize=18)
print(total/ last_x)

x1 = 0
y1 = y2 = (total/ last_x)

x2 = 48

ax.plot([x1, x2], [y1, y2], color='k', linestyle='dotted', linewidth=2)

plt.savefig('project_based_accuracy_evolve.png', dpi=300)


# In[86]:


# x_from = 0
# last_x = None
# fig, ax = plt.subplots(figsize=(16, 6))
# ax.set_xlim(xmax=50)
# total = 0
# for index, project in enumerate(PROJECTS):
#     x = list(range(x_from, x_from + len(F1_HISTORY[project]))) 
#     y = F1_HISTORY[project]
#     total += sum(y)
#     if(last_x):
#         ax.plot([last_x,x[0]],[last_y,y[0]],color=get_color(index))
#     last_x = x[-1]
#     last_y = y[-1]
#     ax.plot(x,y,color=get_color(index), label=project)
#     x_from += len(F1_HISTORY[project])
# plt.legend(loc="lower right", prop={'size': 13})
# plt.ylabel('F1 Score',fontsize=18)
# plt.xlabel('Iteration',fontsize=18)
# x1 = 0
# y1 = y2 = (total/ last_x) + .06
# print(total/ last_x)
# x2 = 48

# ax.plot([x1, x2], [y1, y2], color='k', linestyle='dotted', linewidth=2)
# plt.savefig('project_based_f1_evolve.png', dpi=300)


# In[ ]:





# In[91]:


import graphviz 

dot_data = tree.export_graphviz(MODEL, out_file=None, 
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = graphviz.Source(dot_data)
print(graph)
# print(MODEL.tree_.max_depth)


# In[ ]:




