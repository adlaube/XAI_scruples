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



explainer = LimeTextExplainer(class_names=class_labels)
dataset_features = [x['title'] + x['text'] for x in data]


keys = ['title', 'text']


instance_idx = 0
test_instance = [{x:data[instance_idx][x] for x in keys}]
test_features = test_instance[0]['title'] + test_instance[0]['text']

exp = explainer.explain_instance(test_features,anecdotes_predict,num_features=10,num_samples=100,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_100.html')

exp = explainer.explain_instance(test_features,anecdotes_predict,num_features=10,num_samples=500,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_500.html')

exp = explainer.explain_instance(test_features,anecdotes_predict,num_features=10,num_samples=1000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_1000.html')

exp = explainer.explain_instance(test_features,anecdotes_predict,num_features=10,num_samples=3000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_3000.html')


instance_idx = 10
test_instance = [{x:data[instance_idx][x] for x in keys}]
test_features = test_instance[0]['title'] + test_instance[0]['text']

exp = explainer.explain_instance(test_features,anecdotes_predict,num_features=10,num_samples=100,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_100.html')

exp = explainer.explain_instance(test_features,anecdotes_predict,num_features=10,num_samples=500,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_500.html')

exp = explainer.explain_instance(test_features,anecdotes_predict,num_features=10,num_samples=1000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_1000.html')

exp = explainer.explain_instance(test_features,anecdotes_predict,num_features=10,num_samples=3000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_3000.html')





