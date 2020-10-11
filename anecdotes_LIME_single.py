from lime.lime_text import LimeTextExplainer
import numpy as np
import pickle
import time

import anecdotes_utils


explainer = LimeTextExplainer(class_names=anecdotes_utils.anecdotes_labels)
test_features = anecdotes_utils.get_merged_instance(0)
start = time.time()
exp = explainer.explain_instance(test_features,anecdotes_utils.anecdotes_predict_lime,num_features=10,num_samples=5000,top_labels=5) #LIME only wants one string...
end = time.time()
exp.save_to_file('exp/anecdotes/lime.html')
filehandler = open("exp.obj","wb")
pickle.dump(exp,filehandler)
filehandler.close()
print('execution time: %f',end-start)





