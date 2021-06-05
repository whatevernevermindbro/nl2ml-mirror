from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import completeness_score
import dagshub

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("LABELED_DATA", help="version of the graph you want regex to label your CSV with", type=str)
args = parser.parse_args()

LABELED_DATA = args.LABELED_DATA
df = pd.read_csv(LABELED_DATA)

X = df['code_block']
y = df['graph_vertex_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
number_of_vertices = y.max()

tfidf_vectorizer = TfidfVectorizer()
tfidf_Xtrain = tfidf_vectorizer.fit_transform(X_train)
tfidf_Xtest = tfidf_vectorizer.transform(X_test)

def find_optimal_clusters(train_data, target_data, model, n_clusters):
    history = []
    for k in n_clusters:
        clustering = model.set_params(n_clusters=k).fit(train_data)
        history.append(completeness_score(target_data, clustering.labels_))
    model.set_params(n_clusters=n_clusters[np.argmax(history)])
    print("For model " + str(model))
    print("optimal number of clusters: " + str(n_clusters[np.argmax(history)]))
    print("with score: " +str(np.max(history)))
    print("-------\n")
    return n_clusters[np.argmax(history)], np.max(history)

def experiments(models, train_data, test_data, target_train, target_test, n_clusters):
    opt_clusts = []
    for data, model in zip(train_data, models):
        opt_clust, best_acc = find_optimal_clusters(data, target_train, model, n_clusters)
        opt_clusts.append(opt_clust)
    print("Accuracy on test data")
    best_on_test = []
    for data, model, params in zip(test_data, models, opt_clusts):
        model.set_params(n_clusters=params)
        print("For model " + str(model))
        acc = 0
        if isinstance(model, AgglomerativeClustering):
            acc = (completeness_score(target_test, model.fit_predict(data)))
        else:
            acc = (completeness_score(target_test, model.predict(data)))
        print(acc)
        best_on_test.append(acc)
        print("-------\n")
    return best_on_test, opt_clusts

models = [AgglomerativeClustering(), KMeans(), MiniBatchKMeans()]
train_data = [tfidf_Xtrain.toarray(), tfidf_Xtrain, tfidf_Xtrain]
test_data = [tfidf_Xtest.toarray(), tfidf_Xtest, tfidf_Xtest]
opt_clusts = []
print("For number of clusters from 2 to 100")
experiments(models, train_data, test_data, y_train, y_test, range(2, 101, 2))

print("-------\n\nFor number of clusters around real number of vertices")
best_on_test, optimal_clusters = experiments(models, train_data, 
                                             test_data, y_train, y_test, 
                                             range(number_of_vertices - 5, number_of_vertices + 20, 2))

data_meta = {'DATASET_PATH': LABELED_DATA
                ,'nrows': X.shape[0]
                ,'label': ['-']
                ,'model': ['-']
                ,'script_dir': __file__}
metric_resuts = {'completeness_score_AgglomerativeClustering': best_on_test[0], 
                  'completeness_score_KMeans_results': best_on_test[1], 
                  'completeness_score_MiniBatchKMeans': best_on_test[2]}

AgglomerativeClustering_params = {'completeness_score': optimal_clusters[0]}
KMeans_params = {'completeness_score': optimal_clusters[1]}
MiniBatchKMeans_params = {'completeness_score': optimal_clusters[2]}

with dagshub.dagshub_logger() as logger:
        print("saving the results..")
        logger.log_hyperparams(data_meta)
        logger.log_hyperparams(AgglomerativeClustering_params)
        logger.log_hyperparams(KMeans_params)
        logger.log_hyperparams(MiniBatchKMeans_params)
        logger.log_metrics(metric_resuts)