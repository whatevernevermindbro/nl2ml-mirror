# Source Code Classification
__This is a repo of the Natural Language to Machine Learning (NL2ML) project of the Laboratory of Methods for Big Data Analysis at Higher School of Economics (HSE LAMBDA).__

The project's official __repo__ is stored on GitLab (HSE LAMBDA repository) - https://gitlab.com/lambda-hse/nl2ml \
The project's __full description__ is stored on Notion - https://www.notion.so/NL2ML-Corpus-1ed964c08eb049b383c73b9728c3a231 \
The project's __experiments__ are stored on DAGsHub - https://dagshub.com/levin/source_code_classification

## Project Goals
### Short-Term Goal
To build a model classifying a source code chunk and to specify where the detected class is exactly in the chunk (tag segmentation).
### Long-Term Goal
To build a model generating code by getting a short raw english task in as an input.

## Repository Description
This repository contains instruments which the project's team has been using to label source code chunks with Knowledge Graph vertices and to train models to recognize these vertices in future. By the Knowledge Graph vertices we mean an elementary part of ML-pipeline. The current latest version of the Knowledge Graph contains the following high-level vertices: `['import', 'data_import', 'data_export', 'preprocessing', 'visualization', 'model', 'deep_learning_model', 'train' 'predict']`.

## Data Download
To download the project data and models:
1. Clone this repository
2. Install DVC from https://dvc.org/doc/install
3. Do `dvc pull data` or `dvc pull data`. Note: if you are failing on `dvc pull [folder_to_pull]`, try `dvc pull [folder_to_pull] --jobs 1`

## Contents:
The instruments which we have been using to reach the project goals are: notebooks parsing from Kaggle API and Github API, data preparation, regex-labellig, training models, validation models, model weights/coefficients analysis, errors analysis, synonyms analysis.

__nl2ml_notebook_parser.py__ - a script for parsing Kaggle notebooks and process them to JSON/CSV/Pandas.

__bert_distances.ipynb__ - a notebook with BERT expiremints concerning sense of distance between BERT embeddings where input tokens were tokenized source code chunks.

__bert_classifier.ipynb__ - a notebook with preprocessing and training BERT-pipeline.

__regex.ipynb__ - a notebook with creating labels for code chunks with regex

__logreg_classifier.ipynb__ - a notebook with training logistic regression model on the regex labels with tf-idf and analyzing the outputs

__Comments vs commented code.ipynb__ - a notebook with a model distinguishing NL-comments from commented source code

__github_dataset.ipynb__ - a notebook with opening github_dataset

__predict_tag.ipynb__ - a notebook with predicting class label (tag) with any model

__svm_classifier.ipynb__ - a notebook with training SVM (replaced by _svm_train.py_) and analyzing SVM outputs

__svm_train.py__ - a script for training SVM model

## Conventions:
- Input CSV: encoding='utf-8', sep=',' and CODE_COLUMN has to be == 'code_block' in all input CSVs
- Knowledge Graphs: GRAPH_DIR has to be in the following format: './graph/graph_v{}.txt'.format(GRAPH_VER)