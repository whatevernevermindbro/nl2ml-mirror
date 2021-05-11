from sqlalchemy import create_engine
from tqdm import tqdm
import pandas as pd
import re

def text_preparation4sql(text):
    if isinstance(text, str):
        ftext = re.sub(r'[^\x00-\x7f]', r'', text)
    else:
        ftext = ''
    return f"'`{ftext}`'"


engine = create_engine("mysql+pymysql://root:$a8`k?B2y4nUxX2G@40.119.1.127:32006/nl2ml")

path2data = '../data'
codeblocks_filename = f'{path2data}/competitions4db.csv'
tableToWriteTo = 'good_competitions'


for df in tqdm(pd.read_csv(codeblocks_filename, chunksize=1000)):
    conn = engine.connect()
    df['comp_name'] = df['comp_name'].apply(lambda name: text_preparation4sql(name))
    df['description'] = df['description'].apply(lambda name: text_preparation4sql(name))
    df.to_sql(tableToWriteTo, con=engine, index=False, if_exists='append', chunksize=1000)
    conn.close()





