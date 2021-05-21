from sqlalchemy import create_engine
from tqdm import tqdm
import pandas as pd


def upload_csv(csv_name, table_name, db, path2data='../data', chunksize=1000):
    filepath= f'{path2data}/{csv_name}.csv'
    for df in tqdm(pd.read_csv(filepath, chunksize=chunksize)):
        df.to_sql(table_name, con=db, index=False, if_exists='append', chunksize=1000)

engine = create_engine("mysql+pymysql://root:$a8`k?B2y4nUxX2G@40.119.1.127:32006/nl2ml_test")

upload_csv('competitions_info_cleaned', 'competitions_info_cleaned', engine)





