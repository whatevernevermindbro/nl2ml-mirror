#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv("markup_data.csv")

graph_act = pd.read_csv("actual_graph.csv")


# In[4]:


graph_act['good_names'] = graph_act.graph_vertex  + '.'  + graph_act.graph_vertex_subclass


# In[5]:


columns_names = list(graph_act['good_names'])
columns_names.insert(0, "code_block")


# In[7]:


temp = np.zeros(shape=(df.shape[0],len(columns_names)), dtype='int')
res = pd.DataFrame(temp, columns=columns_names)
res.code_block = df.code_block

for row in range(df.shape[0]):
    res[columns_names[df['graph_vertex_id'][row]]][row] = 1


# In[8]:


res.to_csv('markup_data.csv')


# In[ ]:




