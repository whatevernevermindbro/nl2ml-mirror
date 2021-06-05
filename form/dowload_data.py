from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import date
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
data.to_csv(f'../data/markup_data_{date.today()}.csv', index=False)

sql = '''select id, graph_vertex, graph_vertex_subclass from graph_vertices'''

data = pd.read_sql_query(sql, engine)
print(data.shape)
data.to_csv(f'../data/actual_graph_{date.today()}.csv', index=False)

sql = '''select id, ref_link, comp_name, comp_type, description, metric, datatype, subject, problemtype, insert_ts
from competitions
where metric is not NULL
and metric != 'unkown metric'
and ref_link is not NUll
and comp_type != 'inClass' '''
data = pd.read_sql_query(sql, engine)
print(data.shape)
data.to_csv(f'../data/competitions_{date.today()}.csv', index=False)


