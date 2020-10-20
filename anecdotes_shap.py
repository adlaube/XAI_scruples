from anecdotes_utils import anecdotes_predict_lime, anecdotes_labels, get_merged_instance, anecdotes_exp_dir, anecdotes_dataset
import shap
import numpy as np


instance_idx = 0
test_sample = get_merged_instance(instance_idx,truncate=True)
anecdotes_dataset = np.array(anecdotes_dataset)
#background dataset that can be used as expectation
background = anecdotes_dataset[np.random.choice(len(anecdotes_dataset),100,replace=False)]
explainershap = shap.DeepExplainer(anecdotes_predict_lime,background)
shap_values_matrix = explainershap.shap_values(test_sample)
shap.summary_plot(shap_values_matrix,test_sample)#ä,feature_names=dataset.feature_names)
print('done')
