import numpy as np
import pandas as pd

import utils.preprocessing as preprocessing
import utils.feature_generation as feature_generation


PARTITION_NAME_TEMPLATE = "./data_parts/labeled_comments_partition{}.npy"
PARTITION_COUNT = 2


def preprocess(code_blocks):
    prep_pipeline = [
        preprocessing.trim_symbols,
        preprocessing.single_lines,
        preprocessing.multiple_lines,
        preprocessing.extract_comments,
    ]
    
    for prep_func in prep_pipeline:
        code_blocks = code_blocks.apply(prep_func, axis=1)
    
    comments = []
    for block_comments in code_blocks["comments"]:
        for comment_data in block_comments:
            comments.append(comment_data[1])
    comments = np.array(comments)
    return pd.DataFrame(data=comments.reshape((-1, 1)), columns=["comment"])


def load_code_blocks():
    all_blocks = pd.read_csv("../data/code_blocks_clean.csv")
    all_blocks = all_blocks["code_block"].to_frame()

    comment_blocks_idx = (
        all_blocks["code_block"].str.contains("#") | 
        (all_blocks["code_block"].str.contains("'''") & 
         (all_blocks["code_block"].str.count("'''") % 2 == 0)) |
        (all_blocks["code_block"].str.contains('"""') & 
         (all_blocks["code_block"].str.count('"""') % 2 == 0))
    )
    
    return preprocess(all_blocks[comment_blocks_idx].reset_index())


target = None
for part_id in range(PARTITION_COUNT):
    part_data = np.load(PARTITION_NAME_TEMPLATE.format(part_id))
    if target is None:
        target = part_data
        continue
    labeled_idx = part_data >= 0
    target[labeled_idx] = part_data[labeled_idx]


train_mask = target >= 0
determined_target = target[train_mask]

all_comments = load_code_blocks()
comment_df = feature_generation.preprocess_comments(all_comments)

train_df = comment_df[train_mask]
train_df["is_good_comment"] = determined_target

train_df.to_csv("train.csv")
