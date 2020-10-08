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

def predictor(texts):

        instances=[]
        for text in texts:
                instance = {
                        'title':'',
                        'text':text
                }
                instances.append(instance)
        
        #response = requests.post('https://norms.apps.allenai.org/api/corpus/predict',json=instances)
        response = requests.post('http://127.0.0.1:5050/api/corpus/predict',json=instances)
        response_json = json.loads(response.text) #will throw exception if num of samples is too high
        #calc probabilities by dividing by sum of alphas
        probabilties = [[
                        x['AUTHOR']/sum(x.values()),
                        x['OTHER']/sum(x.values()),
                        x['EVERYBODY']/sum(x.values()),
                        x['NOBODY']/sum(x.values()),
                        x['INFO']/sum(x.values())]
                        for x in response_json
                        ]
        return np.reshape(np.asarray(probabilties),(-1,5))


data = []
data_dir = os.getcwd() + "/data/anecdotes/"
with open(data_dir + 'test.scruples-corpus.jsonl', 'r') as datafile:
    for ln in datafile:
        row = json.loads(ln)
        data.append(row)


class_labels = ["AUTHOR", "OTHER", "EVERYBODY", "NOBODY", "INFO"]
explainer = LimeTextExplainer(class_names=class_labels)
dataset_features = [x['title'] + x['text'] for x in data]


keys = ['title', 'text']


instance_idx = 20
test_instance = [{x:data[instance_idx][x] for x in keys}]
test_features = test_instance[instance_idx]['title'] + test_instance[instance_idx]['text']

exp = explainer.explain_instance(test_features,predictor,num_features=10,num_samples=100,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_100.html')

exp = explainer.explain_instance(test_features,predictor,num_features=10,num_samples=500,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_500.html')

exp = explainer.explain_instance(test_features,predictor,num_features=10,num_samples=1000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_1000.html')

exp = explainer.explain_instance(test_features,predictor,num_features=10,num_samples=3000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_3000.html')


instance_idx = 10
test_instance = [{x:data[instance_idx][x] for x in keys}]
test_features = test_instance[instance_idx]['title'] + test_instance[instance_idx]['text']

exp = explainer.explain_instance(test_features,predictor,num_features=10,num_samples=100,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_100.html')

exp = explainer.explain_instance(test_features,predictor,num_features=10,num_samples=500,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_500.html')

exp = explainer.explain_instance(test_features,predictor,num_features=10,num_samples=1000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_1000.html')

exp = explainer.explain_instance(test_features,predictor,num_features=10,num_samples=3000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_3000.html')





