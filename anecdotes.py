from anecdotes_LIME import explain_anecdote_lime
from anecdotes_utils import anecdotes_data, anecdotes_labels

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter

from bs4 import BeautifulSoup
import subprocess
import pickle

import en_core_web_sm
## spacy for part of speech
nlp = en_core_web_sm.load()

### CONFIGURE

param_dict = {
	"max_number_of_pertubations": 100,
	"adaptive_pertubations": True,
	"number_of_features": 5,
	"number_of_samples": 120
}

FILTER_INCORRECT_PREDICTIONS = True  # will still be stored and plotted, just not considered for the class statistics
LOAD_FROM_PICKLE = True
experiment_from_pickle = pickle.load(open("exp/anecdotes/random120/out.pickle", "rb"))

### CONFIGURE END


def get_explanation(idx: int, param_dict: dict):
	if LOAD_FROM_PICKLE:
		exp = experiment_from_pickle[idx]
	else:
		exp = explain_anecdote_lime(idx, param_dict)
	return exp


def plot_dataset_info(anecdotes_df: pd.DataFrame):
	anecdotes_df["text length"] = anecdotes_df["text"].str.len()
	anecdotes_df.hist(column=["text length"], bins=10)
	# anecdotes_df['text length'].plot(kind='bar')
	plt.title("text length distribution")
	plt.savefig("textlength_hist.png")

	sample_label_df = anecdotes_df["label"].value_counts()
	sample_label_df.plot(kind="bar")
	plt.title("class distribution")
	plt.xticks(rotation=10)
	plt.savefig("label_hist.png")

	sample_type_df = anecdotes_df["post_type"].value_counts()
	# sample_type_df.hist()
	sample_type_df.plot(kind="bar")
	plt.xticks(rotation=10)
	plt.title("type distribution")
	plt.savefig("type_hist.png")
	# label_scores count useful?

	plt.clf()
	# hist with feature scores


if __name__ == "__main__":

	sample_indices = np.random.randint(0, 2500, param_dict['number_of_samples'], np.uint32)
	sample_indices = np.unique(sample_indices)

	if LOAD_FROM_PICKLE:
		sample_indices = list(experiment_from_pickle.keys())

	## META DATAFRAME
	meta_columns_df = ["Date", "Code commit", "Sample selection", "Accuracy"]
	meta_df = pd.DataFrame(columns=meta_columns_df, index=["Test report"])
	meta_df["Date"] = meta_df["Date"].astype(str)
	meta_df["Date"] = [datetime.now().strftime("%d/%m/%Y %H:%M:%S")]
	meta_df["Code commit"] = meta_df["Code commit"].astype(str)
	try:
		meta_df["Code commit"] = subprocess.check_output(["git", "describe"]).strip()
	except:
		pass
	meta_df["Sample selection"] = meta_df["Sample selection"].astype(str)
	meta_df["Sample selection"] = "random"
	meta_df.transpose()

	## MAIN DATAFRAME
	anecdotes_df = pd.DataFrame(anecdotes_data)
	anecdotes_df = anecdotes_df.iloc[sample_indices]
	plot_dataset_info(anecdotes_df)

	## PARAM DATAFRAME
	param_df = pd.DataFrame.from_records(param_dict, index=["Parameter"])
	param_df.transpose()

	#inits
	all_exps = []
	all_exps_dict = {}
	features_per_class = []
	pos_per_class = []
	children_per_class = []
	head_per_class = []
	sentence_per_class = []
	for label in anecdotes_labels:
		features_per_class.append([])
		pos_per_class.append([])
		children_per_class.append([])
		head_per_class.append([])
		sentence_per_class.append([])
		all_exps.append([])
	out_soup = None
	# counter for accuracy
	correct_predictions_cnt = 0

	divider = BeautifulSoup(features="lxml").new_tag("hr")

	outlier_dict = {
		label: {
			"HIGH_ID": 0,
			"HIGH_VALUE": 0,
			"LOW_ID": 0,
			"LOW_VALUE": 0,
		}
		for label in anecdotes_labels
	}

	for idx in sample_indices:
		exp = get_explanation(idx, param_dict)
		all_exps_dict[idx] = exp #save exp into dict
		all_exps[exp.top_labels[0]].append(exp)  # sort per class
		anecdotes_df.loc[idx, "prediction"] = anecdotes_labels[exp.top_labels[0]] #save prediction
		
		# counter for accuracy
		if anecdotes_labels[exp.top_labels[0]] == anecdotes_df.loc[idx]["label"]:
			correct_predictions_cnt += 1
		elif FILTER_INCORRECT_PREDICTIONS: 
			# skip feature processing for wrong predictions
			continue

		# part of speech
		text_annotated = nlp(anecdotes_df.loc[idx]["title"] + ' ' + anecdotes_df.loc[idx]["text"])

		# Process features and contributions per class
		for label_idx in range(len(anecdotes_labels)):
			features_of_one_class_tuple = exp.as_list(label_idx)

			contribution_sum = np.sum(
				[element[1] for element in features_of_one_class_tuple]
			)

			if (
				contribution_sum
				> outlier_dict[anecdotes_labels[label_idx]]["HIGH_VALUE"]
			):
				outlier_dict[anecdotes_labels[label_idx]]["HIGH_ID"] = idx
				outlier_dict[anecdotes_labels[label_idx]][
					"HIGH_VALUE"
				] = contribution_sum

			if (
				contribution_sum
				< outlier_dict[anecdotes_labels[label_idx]]["LOW_VALUE"]
			):
				outlier_dict[anecdotes_labels[label_idx]]["LOW_ID"] = idx
				outlier_dict[anecdotes_labels[label_idx]][
					"LOW_VALUE"
				] = contribution_sum

			# feature_iterator = iter(features_of_one_class_tuple)

			# feature_idx = 0
			# while feature_idx < len(features_of_one_class_tuple):
			# 	feature = features_of_one_class_tuple[feature_idx][0]
			# 	feature_pos_list = []
			# 	feature_child_list = []
			# 	for word in text_annotated:
			# 		if (
			# 			word.text == feature
			# 		):  # iterate over text, for multiple occurences take the last part of speech
			# 			feature_pos_list.append(word.pos_)
			# 			features_of_one_class_tuple[feature_idx][0] += ':' + ','.join([child.text for child in word.children])
						
			# 			continue
			# 	feature_idx+=1
				

			feature_pos_list = []
			feature_child_list = []
			feature_head_list = []
			feature_sentence_list = []
			for feature, contribution in features_of_one_class_tuple:
				processed = False
				children_to_append = ''
				head_to_append = ''
				sentence_to_append = ''				
				for word in text_annotated:

					if (
						word.text == feature
					):  # iterate over text, for multiple occurences take the last part of speech
						if processed == False:
							feature_pos_list.append(word.pos_)
							children_to_append = ','.join([child.text for child in word.children])
							head_to_append = word.head.text
							sentence_to_append = word.sent.text
							processed = True
						else:
							sentence_to_append = 'A! ' + sentence_to_append #mark ambigious
							break #break if ambigious feature

				feature_child_list.append(children_to_append)
				feature_head_list.append(head_to_append)
				feature_sentence_list.append(sentence_to_append)

			#for word in text_annotated:
				#if word.head

			pos_per_class[label_idx] += feature_pos_list
			children_per_class[label_idx] += feature_child_list
			head_per_class[label_idx] += feature_head_list
			sentence_per_class[label_idx] += feature_sentence_list
			features_per_class[label_idx] += features_of_one_class_tuple
			

		# HTML processing
		exp_html = exp.as_html()
		exp_soup = BeautifulSoup(exp_html, features="lxml")
		b_tag = exp_soup.new_tag("b")
		# write index and label
		idx_as_string = str(idx) + " : " + anecdotes_df.loc[idx]["label"]
		b_tag.string = idx_as_string
		exp_soup.body.insert(0, divider)
		exp_soup.body.insert(0, b_tag)
		if out_soup is None:
			out_soup = exp_soup
		else:
			out_soup.body.append(exp_soup.body)

	## add accuracy
	accuracy = correct_predictions_cnt / len(sample_indices) * 100
	meta_df["Accuracy"] = accuracy

	class_soup_list = []
	mean_probabilities = None
	for label_idx in range(len(anecdotes_labels)):
		label = anecdotes_labels[label_idx]
		column_feature = label + " features"
		exp_df = pd.DataFrame(
			features_per_class[label_idx], columns=[column_feature, "contributions"]
		)
		exp_df['sentence'] = sentence_per_class[label_idx]
		exp_df['head'] = head_per_class[label_idx]
		exp_df['children'] = children_per_class[label_idx]
		
		plt.clf()
		exp_df.boxplot(column=["contributions"])
		plt.title(label + " box plot")
		plt.savefig(label + ".png")

		##determine highest contributions
		exp_df = exp_df.iloc[
			exp_df.contributions.abs().sort_values(ascending=False).index
		].reset_index(drop=True)
		topcontributions_df = exp_df.iloc[0:19]  # highest contributions as df

		##determine class probabilities per predicted class
		if not all_exps[
			label_idx
		]:  # needed if there is a class that has not been predicted
			new_probabilities = [0, 0, 0, 0, 0]
		else:
			new_probabilities = np.mean(
				np.vstack([x.predict_proba for x in all_exps[label_idx]]), axis=0
			)
		new_probabilities = np.expand_dims(new_probabilities, axis=0)

		if mean_probabilities is None:
			mean_probabilities = new_probabilities
		else:
			mean_probabilities = np.concatenate([mean_probabilities, new_probabilities])

		##statistics of top features per class
		feature_counter = Counter(exp_df[column_feature])
		features_df = pd.DataFrame()
		for feature, cnt in feature_counter.most_common(5):
			feature_series = exp_df.loc[exp_df[column_feature] == feature][
				"contributions"
			].describe()
			feature_series[column_feature] = feature
			features_df = features_df.append(feature_series, ignore_index=True)

		## switch columns of the feature statistics
		cols = features_df.columns.tolist()
		new_cols = cols.copy()
		new_cols[0] = cols[3]
		new_cols[3] = cols[0]
		features_df = features_df[new_cols]

		## part of speech counting
		pos_counter = Counter(pos_per_class[label_idx])
		pos_df = pd.DataFrame(
			pos_counter.most_common(5), columns=[label + " part of speech", "count"]
		)


		## process html per class
		img_soup = out_soup.new_tag("img", src=label + ".png", alt="hist")
		class_soup_list.append(BeautifulSoup(features_df.to_html(), "html.parser"))
		class_soup_list.append(
			BeautifulSoup(topcontributions_df.to_html(), "html.parser")
		)
		class_soup_list.append(BeautifulSoup(pos_df.to_html(), "html.parser"))
		class_soup_list.append(img_soup)

	prob_df = pd.DataFrame(
		mean_probabilities, columns=anecdotes_labels, index=anecdotes_labels
	)

	confusion_df = pd.crosstab(anecdotes_df["label"], anecdotes_df["prediction"])
	html_confusion = confusion_df.to_html()
	confusion_soup = BeautifulSoup(html_confusion, "html.parser")

	outlier_df = pd.DataFrame(outlier_dict)
	outlier_df = outlier_df.round(2)
	outlier_df = outlier_df.transpose()
	html_outlier = outlier_df.to_html()
	outlier_soup = BeautifulSoup(html_outlier, "html.parser")

	html_string_meta = meta_df.to_html()
	meta_soup = BeautifulSoup(html_string_meta, "html.parser")

	html_string_param = param_df.to_html()
	param_soup = BeautifulSoup(html_string_param, "html.parser")

	html_string_prob = prob_df.to_html()
	prob_soup = BeautifulSoup(html_string_prob, "html.parser")

	# bottom to top
	for soup in class_soup_list:
		out_soup.body.insert(0, divider)
		out_soup.body.insert(0, soup)


	heading = out_soup.new_tag("h2")

	out_soup.body.insert(0, divider)
	out_soup.body.insert(0, prob_soup)
	heading_prob = out_soup.new_tag("h2")
	heading_prob.string = 'Probability distribution per class'
	out_soup.body.insert(0, heading_prob)

	out_soup.body.insert(0, divider)
	out_soup.body.insert(0, confusion_soup)
	heading_conf = out_soup.new_tag("h2")
	heading_conf.string = 'Confusion matrix'
	out_soup.body.insert(0, heading_conf)

	out_soup.body.insert(0, divider)
	out_soup.body.insert(0, outlier_soup)
	heading_out = out_soup.new_tag("h2")
	heading_out.string = 'Outliers: Sum of contributions per class, highest and lowest'
	out_soup.body.insert(0, heading_out)

	img_tag = out_soup.new_tag("img", src="textlength_hist.png", alt="hist")
	out_soup.body.insert(0, img_tag)

	img_tag = out_soup.new_tag("img", src="type_hist.png", alt="hist")
	out_soup.body.insert(0, img_tag)

	img_tag = out_soup.new_tag("img", src="label_hist.png", alt="hist")
	out_soup.body.insert(0, img_tag)

	out_soup.body.insert(0, divider)
	out_soup.body.insert(0, param_soup)
	out_soup.body.insert(0, divider)
	out_soup.body.insert(0, meta_soup)

	# write HTML
	with open("out.html", "w") as file:
		file.write(str(out_soup))

	# write pickle
	if LOAD_FROM_PICKLE == False:
		pickle.dump(all_exps_dict, open("out.pickle", "wb"))
