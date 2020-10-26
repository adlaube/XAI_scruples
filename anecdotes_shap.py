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
        sampler= torch.utils.data.RandomSampler(data_source=dataset,replacement=True,num_samples=50),
        batch_size=50
)

test_loader  = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=5
)

with torch.no_grad():
        # for mb_features in background_loader:
        #     mb_features = {k: v.to(device) for k, v in mb_features.items()}
        #     mb_alphas = torch.exp(model(**mb_features)[0])        
        background_iter = iter(background_loader)
        background_features = next(background_iter) #model expects dict with keys input ids and attention mask, value shapes Nx512
        background_features_list = [background_features['input_ids'].to(device),background_features['attention_mask'].to(device)]

        test_iter = iter(test_loader)
        test_features = next(test_iter)
        test_features_list = [test_features['input_ids'].to(device),test_features['attention_mask'].to(device)]

explainershap = shap.DeepExplainer(model,background_features_list) #shap requires background data to be one tensor only
shap_values_matrix = explainershap.shap_values(test_features_list)
shap.summary_plot(shap_values_matrix)#ä,feature_names=dataset.feature_names)
print('done')
