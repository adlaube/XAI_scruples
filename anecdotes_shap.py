from anecdotes_utils import anecdotes_labels, get_merged_instance, anecdotes_exp_dir, anecdotes_dataset
import shap
import numpy as np
from scruples.demos.norms.app import load_model, get_device
from scruples import settings
from scruples.demos.norms import utils
import torch

model,featurize,delabelize = load_model(dataset='corpus')

device = get_device()

dataset = utils.PredictionDataset(
        features=anecdotes_dataset,
        transform=featurize)

background_loader  = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler= torch.utils.data.RandomSampler(data_source=dataset,replacement=True,num_samples=100),
        batch_size=100
)

with torch.no_grad():
        background_iter = iter(background_loader)
        background_features = next(background_iter) #model expects dict with keys input ids and attention mask, value shapes Nx512


test_sample = featurize(anecdotes_dataset[0])
explainershap = shap.DeepExplainer(model,background_features) #shap requires background data to be one tensor only
shap_values_matrix = explainershap.shap_values(test_sample)
shap.summary_plot(shap_values_matrix,test_sample)#ä,feature_names=dataset.feature_names)
print('done')
