from anchor import anchor_text
from anecdotes_utils import anecdotes_predict_anchor, anecdotes_labels, get_merged_instance, anecdotes_exp_dir
import spacy


instance_idx = 0
nlp = spacy.load('en_core_web_lg')
explainer = anchor_text.AnchorText(nlp, anecdotes_labels, use_unk_distribution=False, use_bert=True)
exp = explainer.explain_instance(get_merged_instance(instance_idx),anecdotes_predict_anchor,threshold=0.95)
exp.save_to_file(anecdotes_exp_dir+"anchor0.html")
print('done')




