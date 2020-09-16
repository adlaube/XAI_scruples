import requests
import json
from lime.lime_text import LimeTextExplainer
from scipy import stats
import os
import numpy as np

from scruples.dataset.readers import ScruplesCorpusDataset

def predictor(texts):

        #response = requests.post('http://127.0.0.1:5050/api/corpus/predict',json=instances)
        
        instances=[]
        for text in texts:
                instance = {
                        'title':'',
                        'text':text
                }
                instances.append(instance)
        
        response = requests.post('https://norms.apps.allenai.org/api/corpus/predict',json=instances)
        response_json = json.loads(response.text)
        #calc probabilities by dividing by sum of alphas
        probabilties = [[
                        x['AUTHOR']/sum(x.values()),
                        x['OTHER']/sum(x.values()),
                        x['EVERYBODY']/sum(x.values()),
                        x['NOBODY']/sum(x.values()),
                        x['INFO']/sum(x.values())]
                        for x in response_json
                        ]
        return np.reshape(np.asarray(probabilties),(5,-1))


data = []
data_dir = os.getcwd() + "/data/anecdotes/"
with open(data_dir + 'test.scruples-corpus.jsonl', 'r') as datafile:
    for ln in datafile:
        row = json.loads(ln)
        data.append(row)
instance_idx = 0
keys = ['title', 'text']
instances = [{x:data[instance_idx][x] for x in keys}]


features = instances[0]['title'] + instances[0]['text']
class_labels = ["AUTHOR", "OTHER", "EVERYBODY", "NOBODY", "INFO"]
explainer = LimeTextExplainer(class_names=class_labels)
exp = explainer.explain_instance(features,predictor,num_features=6, labels=[0, 1],num_samples=5) #LIME only wants one string...
exp.as_list()
print('done')


