# Source Code Classification
This is a repo of NL2ML-project of the Laboratory of Methods for Big Data Analysis at Higher School of Economics (HSE LAMBDA).\
The repo is a mirror of HSE LAMBDA GitLab - https://gitlab.com/lambda-hse/nl2ml \
The project page - https://www.notion.so/NL2ML-Corpus-1ed964c08eb049b383c73b9728c3a231

## Project Goals:

The current short-term goal is to build a model that will be able to classify a source code chunk and to specify where the detected class is exactly in the chunk (tag segmentation).

The main goal is to build a model that will be able to generate code getting a text of the task in english as an input.

## Contents:
__nl2ml_notebook_parser.py__ - a script for parsing Kaggle notebooks and process them to JSON/CSV/Pandas.

__bert_distances.ipynb__ - a notebook with BERT expiremints concerning sense of distance between BERT embeddings where input tokens were tokenized source code chunks.

__bert_classifier.ipynb__ - a notebook with preprocessing and training BERT-pipeline.

__regex.ipynb__ - a notebook with creating labels for code chunks with regex

__logreg_classifier.ipynb.ipynb__ - a notebook with training logistic regression model on the regex labels with tf-idf and analyzing the outputs

__Comments vs commented code.ipynb__ - a notebook with a model distinguishing NL-comments from commented source code

__github_dataset.ipynb__ - a notebook with opening github_dataset

__predict_tag.ipynb__ - a notebook with predicting class label (tag) with any model

__svm_classifier.ipynb__ - a notebook with training SVM (replaced by _svm_train.py_) and analyzing SVM outputs

__svm_train.py__ - a script for training SVM model
