from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
import numpy as np
import pickle
import time


from anecdotes_utils import anecdotes_predict_lime, anecdotes_labels, anecdotes_dataset, get_merged_instance

def explain_anecdote_lime(index: int,num_of_features: int,num_of_pertubations: int = None):
        explainer = LimeTextExplainer(class_names=anecdotes_labels)
        features = get_merged_instance(index)
        anecdote_word_count = len(features)
        if num_of_pertubations is None:
                num_of_pertubations = np.rint(np.min([np.power(anecdote_word_count,1.2),4000]))
                num_of_pertubations = np.int16(num_of_pertubations)

        #start = time.time()
        exp = explainer.explain_instance(features,anecdotes_predict_lime,num_features=num_of_features,
                                                num_samples=num_of_pertubations,top_labels=3) 
        #end = time.time()
        
        exp.save_to_file(str('exp/anecdotes/' + str(index) + '_LIME_' + str(num_of_pertubations) + '_' + str(anecdote_word_count) + '.html'))
        #filehandler = open("exp.obj","wb")
        #pickle.dump(exp,filehandler)
        #filehandler.close()
        #print('execution time': %f',end-start)

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

