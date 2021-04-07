from sqlalchemy import *
from sqlalchemy.types import *
from sqlalchemy.schema import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import func
from sqlalchemy.orm import sessionmaker
import datetime
# from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
import numpy as np

Base = declarative_base()

class Competition(Base):
    __tablename__ = 'competitions'
    id = Column(
        Integer,
        primary_key=True
    )
    ref = Column(
        String,
        nullable=False
    )
    comp_name = Column(
        String
    )
    comp_type = Column(
        String
    )
    description = Column(
        Text(4294000000)
    )
    metric = Column(
        String(1000)
    )
    datatype = Column(
        String(1000)
    )
    subject = Column(
        String(1000)
    )
    problemtype = Column(
        String(1000)
    )
    # notebook = relationship("Notebook", back_populates="competition")
    insert_ts = Column(
        DateTime,
        index=False,
        unique=False,
        nullable=False,
        default=datetime.datetime.utcnow
    )


class Notebook(Base):
    """table with notebooks that weren't started"""

    __tablename__ = 'notebooks'
    id = Column(
        Integer,
        primary_key=True
    )
    kaggle_link = Column(
        String(100),
        nullable=False,
        unique=False
    )
    kaggle_id = Column(
        Integer,
        nullable=False,
        unique=True
    )
    kaggle_score = Column(
        Float(precision=32),
    )
    kaggle_comments = Column(
        Integer
    )
    kaggle_upvotes = Column(
        Integer
    )
    created_on = Column(
        DateTime,
        index=False,
        unique=False,
        nullable=False
    )
    insert_ts = Column(
        DateTime,
        index=False,
        unique=False,
        nullable=False,
        default=datetime.datetime.utcnow
    )
    # chunks = relationship("CodeBlock", back_populates="notebooks")
    competition_id = Column(
        Integer,
        ForeignKey('competitions.id')
    )
    competition = relationship("Competition", back_populates="notebooks")
    # chosen = relationship("ChosenNotebook", back_populates="notebook", uselist=False)
    chosen = Column(
        Integer
    )


Competition.notebooks = relationship('Notebook', order_by=Notebook.id, back_populates="competition")


class Chunk(Base):
    """table with chunks data"""

    __tablename__ = 'chunks'
    id = Column(
        Integer,
        primary_key=True
    )
    code_block_id = Column(
        Integer,
        ForeignKey('codeblocks.id')
    )
    codeblock = relationship("CodeBlock", back_populates="chunks")
    data_format = Column(
        String(100),
        nullable=False,
        unique=False
    )
    # graph_vertex = Column(
    #     String(100),
    #     nullable=False,
    #     unique=False
    # )
    # graph_vertex_subclass = Column(
    #     String(100),
    #     nullable=False,
    #     unique=False
    # )
    graph_vertex_id = Column(
        Integer,
        ForeignKey('graph_vertices.id')
    )
    graph_vertex = relationship('Graph', back_populates="chunks")
    errors = Column(
        String(10),
        nullable=False,
        unique=False
    )
    marks = Column(
        Integer,
        nullable=True,
        unique=False,
    )
    username = Column(
        String(100),
        nullable=False,
        unique=False
    )
    created_on = Column(
        DateTime,
        index=False,
        unique=False,
        nullable=False,
        default=datetime.datetime.utcnow
    )


class Graph(Base):
    __tablename__ = 'graph_vertices'
    id = Column(
        Integer,
        primary_key=True
    )
    graph_vertex = Column(
        String(100),
        nullable=False,
        unique=False
    )
    graph_vertex_subclass = Column(
        String(100),
        nullable=False,
        unique=False
    )
    chunks = relationship('Chunk', back_populates="graph_vertex")
    created_on = Column(
        DateTime,
        index=False,
        unique=False,
        nullable=False,
        default=datetime.datetime.utcnow
    )


class History(Base):
    __tablename__ = 'history'
    id = Column(
        Integer,
        primary_key=True
    )
    code_block_id = Column(
        Integer,
        ForeignKey('codeblocks.id')
    )
    codeblock = relationship("CodeBlock", back_populates="actions")
    username = Column(
        String(100),
        nullable=False,
        unique=False
    )
    created_on = Column(
        DateTime,
        index=False,
        unique=False,
        nullable=False,
        default=datetime.datetime.utcnow
    )


class CodeBlock(Base):
    """table with parsed data"""

    __tablename__ = 'codeblocks'
    id = Column(
        Integer,
        primary_key=True
    )
    code_block = Column(
        Text(4294000000)
    )
    next_code_block_id = Column(
        Integer
    )
    prev_code_block_id = Column(
        Integer
    )
    notebook_id = Column(
        Integer,
        ForeignKey('notebooks.id')
    )
    notebook = relationship("Notebook", back_populates="codeblocks")
    actions = relationship("History", back_populates="codeblock")
    created_on = Column(
        DateTime,
        index=False,
        unique=False,
        nullable=False
    )
    insert_ts = Column(
        DateTime,
        index=False,
        unique=False,
        nullable=False,
        default=datetime.datetime.utcnow
    )


Notebook.codeblocks = relationship('CodeBlock', order_by=CodeBlock.id, back_populates="notebook")
CodeBlock.chunks = relationship('Chunk', order_by=Chunk.id, back_populates="codeblock")


class Data(Base):
    """table with parsed data"""

    __tablename__ = 'chunks_data'
    id = Column(
        Integer,
        primary_key=True
    )
    kaggle_score = Column(
        Float,
    )
    kaggle_comments = Column(
        String(1000)
    )
    kaggle_upvotes = Column(
        Integer
    )
    kaggle_link = Column(
        String(500)
    )
    kaggle_id = Column(
        Integer
    )
    ref = Column(
        String(500)
    )
    data_sources = Column(
        Text(4294000000)
    )
    code_block = Column(
        Text(4294000000)
    )
    code_block_id = Column(
        Integer
    )
    insert_ts = Column(
        DateTime,
        index=False,
        unique=False,
        nullable=False,
        default=datetime.datetime.utcnow
    )