from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
import pickle
import time
from anecdotes_utils import anecdotes_predict_lime, anecdotes_labels, anecdotes_dataset


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





