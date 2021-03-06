import numpy as np
import json
import requests
import os

anecdotes_labels = ["AUTHOR", "OTHER", "EVERYBODY", "NOBODY", "INFO"]


anecdotes_data = []
anecdotes_exp_dir = os.getcwd() + "/data/anecdotes/"
with open(anecdotes_exp_dir + 'test.scruples-corpus.jsonl', 'r') as datafile:
    for ln in datafile:
        row = json.loads(ln)
        anecdotes_data.append(row)

""" Arguments:  - index to instance from dataset
                - truncation if needed, extract head and tail of instance
    Return:     string containing both title and body of the story 
    """

def get_merged_instance(instance_idx,truncate=False):
        keys = ['title', 'text']
        instance = [{x:anecdotes_data[instance_idx][x] for x in keys}]
        merged_instance = instance[0]['title'] + ' ' + instance[0]['text']

        PARAM_MAX_WORD_LENGTH = 50
        split_instance = merged_instance.split(" ")
        if truncate == True:
                total_word_count = len(split_instance)
                if total_word_count > PARAM_MAX_WORD_LENGTH: #poor estimate from word count to tokens, tokens are limited to 512
                        excess_word_count = total_word_count - PARAM_MAX_WORD_LENGTH
                        split_idx_head = np.uint32((total_word_count - excess_word_count) / 2)
                        split_idx_tail = np.uint32((total_word_count + excess_word_count) / 2)
                        truncated_instance = split_instance[:split_idx_head] + split_instance[split_idx_tail:]
                        #join back to string
                        truncated_instance = ' '.join(truncated_instance)
                        return truncated_instance

        return merged_instance


# interface LIME to scruples
def anecdotes_predict_lime(texts):

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
        #calc means of the returned alpha values for the dirichlet distribution
        means = [[
                        x['AUTHOR']/sum(x.values()),
                        x['OTHER']/sum(x.values()),
                        x['EVERYBODY']/sum(x.values()),
                        x['NOBODY']/sum(x.values()),
                        x['INFO']/sum(x.values())]
                        for x in response_json
                        ]
        return np.reshape(np.asarray(means),(-1,5))

# interface anchors to scruples
def anecdotes_predict_anchor(texts):

        instances=[]
        for text in texts:
                instance = {
                        'title':'',
                        'text':text
                }
                instances.append(instance)
        
       # response = requests.post('https://norms.apps.allenai.org/api/corpus/predict',json=instances)
        response = requests.post('http://127.0.0.1:5050/api/corpus/predict',json=instances)
        response_json = json.loads(response.text) #will throw exception if num of samples is too high
        #prediction is the label with the highest alpha score
        predicted_labels = [max(response_dict, key=response_dict.get)
                        for response_dict in response_json
                        ]



        return np.array(predicted_labels)

