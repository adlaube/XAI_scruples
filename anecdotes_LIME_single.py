import requests
import json
from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
from scipy import stats
import os
import numpy as np
import pickle
import time


from scruples.dataset.readers import ScruplesCorpusDataset
from anecdotes_utils import anecdotes_predict, class_labels

data = []
data_dir = os.getcwd() + "/data/anecdotes/"
with open(data_dir + 'test.scruples-corpus.jsonl', 'r') as datafile:
    for ln in datafile:
        row = json.loads(ln)
        data.append(row)

instance_idx = 0
keys = ['title', 'text']
test_instance = [{x:data[instance_idx][x] for x in keys}]
test_features = test_instance[0]['title'] + test_instance[0]['text']

dataset_features = [x['title'] + x['text'] for x in data]

explainer = LimeTextExplainer(class_names=class_labels)
start = time.time()
exp = explainer.explain_instance(test_features,anecdotes_predict,num_features=10,num_samples=5000,top_labels=5) #LIME only wants one string...
end = time.time()
exp.save_to_file('exp/anecdotes/lime.html')
filehandler = open("exp.obj","wb")
pickle.dump(exp,filehandler)
filehandler.close()
print('execution time: %f',end-start)





