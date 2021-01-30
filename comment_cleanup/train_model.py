import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from scipy.special import softmax

import os

import warnings
warnings.simplefilter('ignore')

from simpletransformers.classification import ClassificationModel

DATASET = './train.csv'
MODEL_FOLDER = './outputs/'
RESULT_FILE = './metrics.txt'

train_data = pd.read_csv(DATASET)

str_train_data = train_data[train_data.columns[1:]]
str_train_data['comment'] = str_train_data['comment'].astype('string')
str_train_data = str_train_data.rename(columns={"comment": "text", 'is_good_comment': "labels"})

str_train_y = str_train_data[str_train_data.columns[-1]]
str_train_x = str_train_data[str_train_data.columns[:-1]]

str_train, str_test = train_test_split(str_train_data, test_size=0.33, random_state=42)

model = ClassificationModel('roberta', 'roberta-base', use_cuda=False,num_labels=2, args={
                                                                     'reprocess_input_data': True,
                                                                     'overwrite_output_dir': True,
                                                                     'fp16': False,
                                                                     'do_lower_case': False,
                                                                     'num_train_epochs': 4,
                                                                     'regression': False,
                                                                     'manual_seed': 42,
                                                                     "learning_rate":2e-5,
                                                                     'weight_decay':0,
                                                                     "save_eval_checkpoints": True,
                                                                     "save_model_every_epoch": False,
                                                                     "silent": True,
                                                                     "output_dir": MODEL_FOLDER})
model.train_model(str_train)

prediction_tra = model.eval_model(str_test)[1]

vals = softmax(prediction_tra,axis=1)

f = open(RESULT_FILE, "w")
f.write(f"Log_Loss: {log_loss(str_test['labels'], vals)}")

dummy_classes = np.zeros(vals.shape[0])
for i in range(vals.shape[0]):
    if vals[i][0] < vals[i][1]:
        dummy_classes[i] = 1

re_cla = str_test['labels'].tolist()
acc = accuracy_score(str_test['labels'], dummy_classes)
f1 = f1_score(str_test['labels'], dummy_classes)#, average='weighted')

f.write('f1 score: ' + str(f1))
f.write('accuracy: ' + str(acc))

average_precision = average_precision_score(str_test['labels'], dummy_classes)

f.write('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
f.close()