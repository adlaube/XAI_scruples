from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
import numpy as np
import pickle
import time


from anecdotes_utils import anecdotes_predict_lime, anecdotes_labels, anecdotes_data, get_merged_instance

def explain_anecdote_lime(index: int,param_dict: dict):
        explainer = LimeTextExplainer(class_names=anecdotes_labels)
        features = get_merged_instance(index)
        anecdote_word_count = len(features)
        if param_dict['adaptive_pertubations'] is True:
                num_of_pertubations = np.rint(np.min([np.power(anecdote_word_count,1.2),param_dict['max_number_of_pertubations']]))
                num_of_pertubations = np.int16(num_of_pertubations)
        else:
                num_of_pertubations = param_dict['max_number_of_pertubations']

        #start = time.time()
        exp = explainer.explain_instance(features,anecdotes_predict_lime,num_features=param_dict['number_of_features'],
                                                num_samples=num_of_pertubations) 
        #end = time.time()
        
        #exp.save_to_file(str('exp/anecdotes/' + str(index) + '_LIME_' + str(num_of_pertubations) + '_' + str(anecdote_word_count) + '.html'))
        #filehandler = open("exp.obj","wb")
        #pickle.dump(exp,filehandler)
        #filehandler.close()
        #print('execution time': %f',end-start)
        return exp

def explain_anecdotes_lime_sp():
        explainer = LimeTextExplainer(class_names=anecdotes_labels)
        #Problem: 5000 perturbations per sample, API muss umgangen werden um das umzustellen
        start = time.time()
        sp_obj = submodular_pick.SubmodularPick(explainer, anecdotes_dataset, anecdotes_predict_lime, sample_size=1, num_features=5,num_exps_desired=10)
        end = time.time()
        for idx,exp_item in enumerate(sp_obj.explanations):
                exp_item.save_to_file("exp/anecdotes/ANEC_" + str(idx) + ".html")
        filehandler = open("exp/anecdotes/SP.obj","wb")
        pickle.dump(sp_obj,filehandler)
        filehandler.close()
        print('execution time: ',end-start)        

