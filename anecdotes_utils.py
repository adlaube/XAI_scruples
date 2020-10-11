import numpy as np
import json
import requests
import os

anecdotes_labels = ["AUTHOR", "OTHER", "EVERYBODY", "NOBODY", "INFO"]


data = []
anecdotes_exp_dir = os.getcwd() + "/data/anecdotes/"
with open(anecdotes_exp_dir + 'test.scruples-corpus.jsonl', 'r') as datafile:
    for ln in datafile:
        row = json.loads(ln)
        data.append(row)

anecdotes_dataset = [x['title'] + x['text'] for x in data]


""" Argument: index to instance from dataset
    Return:     string containing both title and body of the story """

def get_merged_instance(instance_idx):
        keys = ['title', 'text']
        instance = [{x:data[instance_idx][x] for x in keys}]
        test_features = instance[0]['title'] + instance[0]['text']
        return test_features

def anecdotes_predict(texts):

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