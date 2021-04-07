from sqlalchemy.orm import sessionmaker
import pandas as pd
import sqlalchemy

# TODO config with paths and passwords
engine = sqlalchemy.create_engine("mysql+pymysql://root:$a8`k?B2y4nUxX2G@40.119.1.127:32006/nl2ml")
Session = sessionmaker(bind=engine)
session = Session()

sql = '''select 
            code_block_id,
            code_block ,
            data_format ,
            graph_vertex_id ,
            errors,
            marks,
            kaggle_id,
            competition_id
from chunks ch 
left join codeblocks c on ch.code_block_id  = c.id 
left join notebooks n on c.notebook_id = n.id'''

data = pd.read_sql_query(sql, engine)
print(data.shape)
data.to_csv(f'../data/markup_data.csv')

