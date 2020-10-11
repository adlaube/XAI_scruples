from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
from scipy import stats
import os
import numpy as np
import pickle
import time
import json
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
#exp = explainer.explain_instance(test_features,predictor,num_features=2,num_samples=10,top_labels=5) #LIME only wants one string...
#exp.save_to_file('lime.html')

#Problem: 5000 perturbations per sample, API muss umgangen werden um das umzustellen
start = time.time()
sp_obj = submodular_pick.SubmodularPick(explainer, dataset_features, anecdotes_predict, sample_size=1, num_features=5,num_exps_desired=10)
end = time.time()
for idx,exp_item in enumerate(sp_obj.explanations):
        exp_item.save_to_file("exp/anecdotes/ANEC_" + str(idx) + ".html")
filehandler = open("exp/anecdotes/SP.obj","wb")
pickle.dump(sp_obj,filehandler)
filehandler.close()
print('execution time: ',end-start)





