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

instance_idx = 0
keys = ['title', 'text']
test_instance = [{x:data[instance_idx][x] for x in keys}]
test_features = test_instance[0]['title'] + test_instance[0]['text']

dataset_features = [x['title'] + x['text'] for x in data]

class_labels = ["AUTHOR", "OTHER", "EVERYBODY", "NOBODY", "INFO"]
explainer = LimeTextExplainer(class_names=class_labels)
#exp = explainer.explain_instance(test_features,predictor,num_features=2,num_samples=10,top_labels=5) #LIME only wants one string...
#exp.save_to_file('lime.html')

#Problem: 5000 perturbations per sample, API muss umgangen werden um das umzustellen
start = time.time()
sp_obj = submodular_pick.SubmodularPick(explainer, dataset_features, predictor, sample_size=1, num_features=5,num_exps_desired=10)
end = time.time()
for idx,exp_item in enumerate(sp_obj.explanations):
        exp_item.save_to_file("exp/anecdotes/ANEC_" + str(idx) + ".html")
filehandler = open("exp/anecdotes/SP.obj","wb")
pickle.dump(sp_obj,filehandler)
filehandler.close()
print('execution time: ',end-start)





