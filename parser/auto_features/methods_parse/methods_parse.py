#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import ast
import traceback


def UsedMethods2(code):
  try:
    p = ast.parse(code.strip('`'))
    names = sorted({node.attr for node in ast.walk(p) if isinstance(node, ast.Attribute)}) #but here we also have node.id for example
    result = list()
    for instance in names:
     pos = code.find(instance)
     print(pos, instance)
     if (pos > 0 and code[pos-1] == '.') and (pos < len(code) - len(instance) and (code[pos + len(instance)] == '.' or code[pos + len(instance) ] == '(')) :
      result.append(instance)
    return result
  except:
    return list()


# attributes
    
def UsedMethods(code):
    try:
        p = ast.parse(code.strip('`'))
        names = sorted({node.attr for node in ast.walk(p) if isinstance(node, ast.Attribute)})
        return names
    except:
        names = list()
        dot_pos = code.find('.')
        
        while dot_pos != -1:
            dot_pos += 1
            name = ""
            while (dot_pos < len(code)) and (code[dot_pos].isalpha() or code[dot_pos].isdigit() or code[dot_pos] in {'(', '.'}):
                if code[dot_pos] in {'(', '.'}:
                    names.append(name)
                    break
                name += code[dot_pos]
                dot_pos += 1

            dot_pos = code.find('.', dot_pos)  
    
    return names
    

#load dataframe    
# filename = "11.csv" #enter the name of a .csv file
# df = pd.read_csv(filename)


# #chop off the head 
# df.columns = df.iloc[0,:]
# df = df.drop(index=0)

#fill methods for each block

# df['python_methods'] = df['code_block'].apply(UsedMethods)


#fill methods for nearrby blocks
def shift_methods(df, shift_range):
    """
    # df['python_methods_p1'] = np.NaN
    # df['python_methods_p2'] = np.NaN
    # df['python_methods_p3'] = np.NaN
    # df['python_methods_m1'] = np.NaN
    # df['python_methods_m2'] = np.NaN
    # df['python_methods_m3'] = np.NaN

    for i in range(1, shift_range + 1):
        df['python_methods_m{}'.format(i)] = np.NaN
        df['python_methods_p{}'.format(i)] = np.NaN
        
        
        # df['graph_vertex_m{}'.format(i)] = np.NaN
        # df['graph_vertex_p{}'.format(i)] = np.NaN
        
        df['python_methods_m{}'.format(i)][i:] = df['python_methods'][:-i]
        df['python_methods_p{}'.format(i)][:-i] = df['python_methods'][i:]
        # df['graph_vertex_m{}'.format(i)][i:] = df['graph_vertex_m'][:-i]
        # df['graph_vertex_p{}'.format(i)][:-i] = df['graph_vertex_p'][i:]

    # df['python_methods_m1'][1:] = df['python_methods'][:-1]
    # df['python_methods_m2'][2:] = df['python_methods'][:-2]
    # df['python_methods_m3'][3:] = df['python_methods'][:-3]
    # df['python_methods_p1'][:-1] = df['python_methods'][1:]
    # df['python_methods_p2'][:-2] = df['python_methods'][2:]
    # df['python_methods_p3'][:-3] = df['python_methods'][3:]
    """
    for i in range(1, shift_range + 1):
        df['python_methods_m{}'.format(i)] = np.NaN
        df['graph_vertex_m{}'.format(i)] = np.NaN
    # for i in range(1, shift_range + 1):
        df['python_methods_p{}'.format(i)] = np.NaN
        df['graph_vertex_p{}'.format(i)] = np.NaN

    nb_list = df['kaggle_id'].unique()
    
    for name in nb_list: 
        df_name = df[df['kaggle_id'] == name]
    
        for i in range(1, min(shift_range + 1, df_name.shape[0])):
            df_name['python_methods_m{}'.format(i)][i:] = df_name['python_methods'][:-i]
            df_name['python_methods_p{}'.format(i)][:-i] = df_name['python_methods'][i:]
            df_name['python_methods_m{}'.format(i)][i:] = df_name['python_methods'][:-i]
            df_name['python_methods_p{}'.format(i)][:-i] = df_name['python_methods'][i:]

            df_name['graph_vertex_m{}'.format(i)][i:] = df_name['graph_vertex'][:-i]
            df_name['graph_vertex_p{}'.format(i)][:-i] = df_name['graph_vertex'][i:]
            df_name['graph_vertex_m{}'.format(i)][i:] = df_name['graph_vertex'][:-i]
            df_name['graph_vertex_p{}'.format(i)][:-i] = df_name['graph_vertex'][i:]

        df[df['kaggle_id'] == name] = df_name
    
    return df


# #export dataframe
# df.to_csv(filename[:-4] + "python_methods.csv", index=False)


