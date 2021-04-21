import numpy as np
import pandas as pd
import ast



def UsedMethods2(code):
  try:
    p = ast.parse(code)
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
  except Exception as e:
    # print(e)
    return list()
    

#load dataframe    
# filename = "11.csv" #enter the name of a .csv file
# df = pd.read_csv(filename)


# #chop off the head 
# df.columns = df.iloc[0,:]
# df = df.drop(index=0)

#fill methods for each block

# df['python_methods'] = df['code_block'].apply(UsedMethods)


#fill methods for nearrby blocks
def shift_methods(df):
    df['python_methods_p1'] = np.NaN
    df['python_methods_p2'] = np.NaN
    df['python_methods_p3'] = np.NaN
    df['python_methods_m1'] = np.NaN
    df['python_methods_m2'] = np.NaN
    df['python_methods_m3'] = np.NaN

    df['python_methods_m1'][1:] = df['python_methods'][:-1]
    df['python_methods_m2'][2:] = df['python_methods'][:-2]
    df['python_methods_m3'][3:] = df['python_methods'][:-3]
    df['python_methods_p1'][:-1] = df['python_methods'][1:]
    df['python_methods_p2'][:-2] = df['python_methods'][2:]
    df['python_methods_p3'][:-3] = df['python_methods'][3:]
    return df


# #export dataframe
# df.to_csv(filename[:-4] + "python_methods.csv", index=False)

