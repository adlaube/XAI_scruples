from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
import pickle

from anecdotes_utils import anecdotes_predict_lime, anecdotes_labels, anecdotes_dataset, get_merged_instance


explainer = LimeTextExplainer(class_names=anecdotes_labels)
test_features = get_merged_instance(20)

exp = explainer.explain_instance(test_features,anecdotes_predict_lime,num_features=10,num_samples=100,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_100.html')

exp = explainer.explain_instance(test_features,anecdotes_predict_lime,num_features=10,num_samples=500,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_500.html')

exp = explainer.explain_instance(test_features,anecdotes_predict_lime,num_features=10,num_samples=1000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_1000.html')

exp = explainer.explain_instance(test_features,anecdotes_predict_lime,num_features=10,num_samples=3000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_20_3000.html')


test_features = get_merged_instance(10)

exp = explainer.explain_instance(test_features,anecdotes_predict_lime,num_features=10,num_samples=100,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_100.html')

exp = explainer.explain_instance(test_features,anecdotes_predict_lime,num_features=10,num_samples=500,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_500.html')

exp = explainer.explain_instance(test_features,anecdotes_predict_lime,num_features=10,num_samples=1000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_1000.html')

exp = explainer.explain_instance(test_features,anecdotes_predict_lime,num_features=10,num_samples=3000,top_labels=3) #LIME only wants one string...
exp.save_to_file('exp/anecdotes/test_lime_10_3000.html')





