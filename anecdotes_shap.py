from anecdotes_utils import anecdotes_labels, get_merged_instance, anecdotes_exp_dir, anecdotes_dataset
import shap
import numpy as np
from scruples.demos.norms.app import load_model
import torch

model,featurize,delabelize = load_model(dataset='corpus')


# dataset = utils.PredictionDataset(
#         features=anecdotes_dataset,
#         transform=featurize)
# data_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=settings.NORMS_PREDICT_BATCH_SIZE,
#         shuffle=False)

anecdotes_array = np.array(anecdotes_dataset)
anecdotes_array = anecdotes_array[np.random.choice(len(anecdotes_dataset),100,replace=False)]#cast to array to perform array indexing
anecdotes_list = anecdotes_array.tolist()


featured_list = []
for idx, anecdote in enumerate(anecdotes_list):
        featured_list.append(featurize(anecdote))

test_sample = featurize(anecdotes_dataset[0])
explainershap = shap.DeepExplainer(model,featured_list)
shap_values_matrix = explainershap.shap_values(test_sample)
shap.summary_plot(shap_values_matrix,test_sample)#ä,feature_names=dataset.feature_names)
print('done')
