#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
# import magic
import sys
from comment_parser import comment_parser

filename = sys.argv[1]
save_as = sys.argv[2]

df = pd.read_csv(filename, sep=',')
df.head(5)


# Save the column names from the original markup file

HEADER = df.columns
HEADER = [col if ("Unnamed:" not in col) else "" for col in HEADER ]

# df.columns = df.iloc[0,:]
# df = df.drop(index=0)

# df.head(5)
graph_path = '../../data/actual_graph_2021-04-18.csv'
graph = pd.read_csv(graph_path)
df = df.merge(graph, left_on='graph_vertex_id', right_on='id', how='left')

def get_dict(str_arr):
    name_lib_import = {}
    name_lib_from = {}
    str_arr = [[line.strip() for line in text.split("\n")] for text in str_arr]
    # delete all double spaces in all lines
    for line_array in str_arr:
        for ind in range(len(line_array)):
            prev_len = -1
            while prev_len != len(line_array[ind]):
                prev_len = len(line_array[ind])
                line_array[ind] = line_array[ind].replace("  ", " ")
    # now get libraries if line start with "import" or "from"
    for line_array in str_arr:
        for line in line_array:
            if line.startswith("import"):
                split_arr = line.split(" ")
                # if we have "import ... as ...""
                if len(split_arr) > 2 and split_arr[2] == "as":
                    name_lib_import[split_arr[-1]] = split_arr[1].split(".")[0]
            
            if line.startswith("from"):
                split_arr = line.split(" ")
                # if we have "from ... import ...""
                if len(split_arr) > 2 and split_arr[2] == "import":
                    name_lib_from[split_arr[-1]] = split_arr[1].split(".")[0]
    return (name_lib_import, name_lib_from)
    
    
def get_libraries(text: str, dicts: tuple):
    name_lib_import = dicts[0]
    name_lib_from = dicts[1]
    text = str(text)
    libs = []
    line_array = [line.strip() for line in text.split("\n")]
    # delete all double spaces in all lines
    for ind in range(len(line_array)):
        
        prev_len = -1
        while prev_len != len(line_array[ind]):
            prev_len = len(line_array[ind])
            line_array[ind] = line_array[ind].replace("  ", " ")
    # now get libraries if line start with "import" or "from"
    for line in line_array:
        if line.startswith("import") or line.startswith("from"):
            libs.append(line.split(" ")[1].split(".")[0])
    for name in name_lib_from.keys():
        for line in line_array:
            if name in line:
                libs.append(name_lib_from[name])
                
    for name in name_lib_import.keys():
        for line in line_array:
            if name + "." in line:
                libs.append(name_lib_import[name])
        
    return "\n".join(list(set(libs)))


def get_comments(text: str):
    text = str(text)
#     comments = comment_parser.extract_comments_from_str(text, mime="text/x-python")
#     return "\n".join([line.text() for line in comments])

    comments = []
    line_array = [line.strip() for line in text.split("\n")]
    # now get libraries if line start with "#"
    for line_ind in range(len(line_array)):
        if line_array[line_ind].startswith("#"):
            comments.append(line_array[line_ind][1:])
        elif line_array[line_ind].startswith("'''"):
            multi_comm = str()
            multi_comm += line_array[line_ind][3:] + "\n"
            if "'''" in multi_comm:
                multi_comm = multi_comm.replace("'''", "")
                line_ind += 1
                comments.append(multi_comm)
                continue
            line_ind += 1
            while line_ind < len(line_array):
                multi_comm += line_array[line_ind] + "\n"
                if "'''" in multi_comm:
                    multi_comm = multi_comm.replace("'''", "")
                    line_ind += 1
                    break
                line_ind += 1
            comments.append(multi_comm)
        elif line_array[line_ind].startswith('"""'):
            multi_comm = str()
            multi_comm += line_array[line_ind][3:] + "\n"
            if '"""' in multi_comm:
                multi_comm = multi_comm.replace('"""', "")
                line_ind += 1
                comments.append(multi_comm)
                continue
            line_ind += 1
            while line_ind < len(line_array):
                multi_comm += line_array[line_ind] + "\n"
                if '"""' in multi_comm:
                    multi_comm = multi_comm.replace('"""', "")
                    line_ind += 1
                    break
                line_ind += 1
            comments.append(multi_comm)
            pass
    return "\n".join([line for line in comments])

df['comments'] = df['code_block'].apply(get_comments)

# For each 'kaggle_id' do cell shift
notebooks_ids = set([i for i in df['kaggle_id'].values if str(i) != 'nan'])
SHIFT_RANGE = 3
all_temp_dfs = []

# print(notebooks_ids)
for not_id in notebooks_ids:
    print('notebook id {}'.format(not_id))
    # get rows for one kaggle_id
    temp_df = df[df['kaggle_id'] == not_id]
    buf_graph_vertex = list(temp_df.graph_vertex.values)
    dicts = get_dict(temp_df.code_block.values)
#     print(dicts)
    buf_arr = []
    for code in temp_df.code_block.values:
#         print(get_libraries(code, dicts))
        buf_arr.append(get_libraries(code, dicts))
    temp_df['libraries'] = buf_arr
    # shift down
    for i in range(1, SHIFT_RANGE + 1):
        print(i, len([np.NaN] * i + buf_graph_vertex[0: -i]))
        print([np.NaN] * i + buf_graph_vertex[0: -i])
        temp_df['graph_vertex_m' + str(i)] = [np.NaN] * i + buf_graph_vertex[0: -i]
    # shift up
    for i in range(1, SHIFT_RANGE + 1):
        temp_df['graph_vertex_p' + str(i)] = buf_graph_vertex[i:] + [np.NaN] * i
    all_temp_dfs.append(temp_df)
    
# concatenate all shifted notebooks
final_df = pd.concat(all_temp_dfs)
final_df['kaggle_id'] = final_df['kaggle_id'].apply(int)
final_df['chunk_id'] = final_df['chunk_id'].apply(int)

final_df.sort_values(['kaggle_id', 'chunk_id'], inplace=True)
final_df.head()


# Returning the original header markup
# add columns names
final_df.loc[-1] = final_df.columns
# add original HEADER
final_df.columns = HEADER
final_df.index = final_df.index + 1
final_df.sort_index(inplace=True)

# final_df.head()
final_df.to_csv(save_as, index=False)
