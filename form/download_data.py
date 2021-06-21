from sqlalchemy.orm import sessionmaker
import pandas as pd
import sqlalchemy
from datetime import date
from getpass import getpass
import yaml

def read_yaml(path):
    with open(path) as f:
        data = yaml.full_load(f)
    return data

def create_engine_link():
    config = read_yaml('../db.yml')
    return f"mysql+pymysql://{config.get('user')}:{config.get('password')}@40.119.1.127:32006/nl2ml"

# TODO config with paths and passwords
engine = sqlalchemy.create_engine(create_engine_link())
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
            competition_id,
            ch.username,
            ch.created_on
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
and ref_link is not NUll '''
data = pd.read_sql_query(sql, engine)
print(data.shape)
data.to_csv(f'../data/competitions_{date.today()}.csv', index=False)


sql = '''
select 
            t1.id as code_block_id,
            code_block ,
            kaggle_id,
            competition_id
from
(select * from codeblocks c where id not in (select DISTINCT code_block_id from chunks c2)) t1
left join notebooks n on t1.notebook_id = n.id 
'''
data = pd.read_sql_query(sql, engine)
print(data.shape)
data.to_csv(f'../data/not_yet_markup_data_{date.today()}.csv', index=False)
