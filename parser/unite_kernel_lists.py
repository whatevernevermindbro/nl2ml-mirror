import os

import pandas as pd

LISTS_FOLDER = "./kernel_lists"
OLD_LIST = "../data/additional_kernels.csv"

all_refs = []
for filename in os.listdir(LISTS_FOLDER):
    path = os.path.join(LISTS_FOLDER, filename)
    with open(path, "r") as f:
        if f.readline().startswith("No kernels found"):
            continue
    df = pd.read_csv(path)
    all_refs.append(df.ref)

df = pd.read_csv(OLD_LIST)
all_refs.append(df.ref)

all_refs = pd.concat(all_refs, ignore_index=True)
df_new = pd.DataFrame()
df_new["ref"] = all_refs.drop_duplicates()
df_new = df_new.reset_index(drop=True)

df_new.to_csv(OLD_LIST)

print(df_new.shape)

for filename in os.listdir(LISTS_FOLDER):
    path = os.path.join(LISTS_FOLDER, filename)
    os.remove(path)
