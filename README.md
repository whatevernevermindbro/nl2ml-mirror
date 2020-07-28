# Source Code Classification
This is an old repo of NL2ML-project of the Laboratory of Big Data Analysis of Higher School of Economics (HSE LAMBDA).

The project page - https://www.notion.so/NL2ML-Corpus-1ed964c08eb049b383c73b9728c3a231

The repo is currently migrating to the HSE LAMBDA GitLab - https://gitlab.com/lambda-hse/nl2ml

## Project Goals:

The current short-term goal is to build a model that will be able to classify a source code chunk and to specify where the detected class is exactly in the chunk (tag segmentation).

The global goal is to build a model that will be able to generate code using a text of the task in english.

## Contents:
__nl2ml_notebook_parser.py__ - script for parsing Kaggle notebooks and process them to JSON/CSV/Pandas.

__bert_distances.ipynb__ - notebook with expiremints concerning sense of distance between BERT embeddings where input tokens were tokenized source code chunks.

__bert_classifier.ipynb__ - notebook with preprocessing and training pipeline.

__regex.ipynb__ - notebook with creating labels for code chunks with regex

__logreg_classifier.ipynb.ipynb__ - notebook with building logreg on the regex labels with tf-idf
