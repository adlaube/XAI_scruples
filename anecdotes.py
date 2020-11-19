from anecdotes_LIME import explain_anecdote_lime
from anecdotes_utils import anecdotes_data,anecdotes_labels

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter

from bs4 import BeautifulSoup

if __name__ == "__main__":

        #np.random.randint(0,2500,50,np.uint32)
        sample_indices = np.array([1491, 2291, 1636,  146,  204, 2092, 1138,  319,  645,  315, 2243,
                                 2357, 1616, 1853,  671,  762, 1240, 1675,  944,  220, 2304,  567,
                                         434,  946,  220, 1317,  877, 2077, 1559, 1509,  439,  670, 2470,
                                         203, 1949,  138, 2437,  506,  401, 2212, 1652,  807,  221,  110,
                                 2478,  567,  551,   69, 1911, 1851],dtype=np.uint32)

        sample_indices = np.array(range(20))
        sample_indices = np.sort(sample_indices)

        ## META DATAFRAME
        meta_columns_df = ['Date','Code commit', 'Sample selection' ]
        meta_df = pd.DataFrame(columns=meta_columns_df)
        meta_df['Date'] = meta_df['Date'].astype(str)
        meta_df['Date'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        ## MAIN DATAFRAME
        anecdotes_df = pd.DataFrame(anecdotes_data)
        anecdotes_df = anecdotes_df.iloc[sample_indices]

        anecdotes_df['text length'] = anecdotes_df['text'].str.len()
        anecdotes_df.hist(column=['text length'],bins = 10)
        #anecdotes_df['text length'].plot(kind='bar')
        plt.title('text length distribution')
        plt.savefig('textlength_hist.png')

        sample_label_df = anecdotes_df['label'].value_counts()
        sample_label_df.plot(kind='bar')
        plt.title('class distribution')
        plt.xticks(rotation=10)
        plt.savefig('label_hist.png')

        sample_type_df = anecdotes_df['post_type'].value_counts()
        #sample_type_df.hist()
        sample_type_df.plot(kind='bar')
        plt.xticks(rotation=10)
        plt.title('type distribution')
        plt.savefig('type_hist.png')
        #label_scores count useful?

        #hist with feature scores

        param_dict = {"max_number_of_pertubations"      :10,
                        "adaptive_pertubations"         :False,
                        "number_of_features"            :10,
                        }
        param_df = pd.DataFrame.from_records(param_dict,index=["Parameter"])
        param_df.transpose()

        out_soup = None

        #init dataframes
        
        explanation_dataframes_dict = {}

        all_exps = []
        features_per_class = []
        for label in anecdotes_labels:
                features_per_class.append([])
                all_exps.append([])

        for idx in sample_indices:

                exp = explain_anecdote_lime(idx,param_dict)
                all_exps[exp.top_labels[0]].append(exp) # sort per class
                # Process features and contributions per class
                for label_idx in range(len(anecdotes_labels)):
                        features_of_one_class_tuple = exp.as_list(label_idx)
                        features_per_class[label_idx] += (features_of_one_class_tuple)
                
                # HTML processing
                exp_html = exp.as_html()
                exp_soup = BeautifulSoup(exp_html,features="lxml")
                b_tag = exp_soup.new_tag('b')
                idx_as_string = str(idx)                
                b_tag.string = idx_as_string
                exp_soup.body.insert(0,b_tag)
                if out_soup is None:    
                        out_soup = exp_soup
                else:
                        out_soup.body.append(exp_soup.body) 

        # hist of top features + contributions
        # feature categories hist
        # statistics of contributions
        # average prediction score as matrix CxC
        # mean variance of feature positions (to indiciate where context matters), search through anecdote necessary
        #compile dataframe per class
        columns_features_df = ["features","contributions"]
        
        columns_main_df = ["class","top features"]
        main_df = pd.DataFrame(columns=columns_main_df)
        main_df['class'] = main_df['class'].astype(str)
        main_df['top features'] = main_df['top features'].astype(object)


        class_soup_list = []
        mean_probabilities = None
        for label_idx in range(len(anecdotes_labels)):
                exp_df = pd.DataFrame(features_per_class[label_idx],columns=columns_features_df)
                label = anecdotes_labels[label_idx]
                explanation_dataframes_dict[label] = exp_df
                feature_counter = Counter(exp_df['features'])
                
                statistics = exp_df['contributions'].describe()
                statistics.round(2)
                if not all_exps[label_idx]:  #needed if there is a class that has not been predicted
                        new_probabilities = [0,0,0,0,0]
                else:
                        new_probabilities = np.mean(np.vstack([x.predict_proba for x in all_exps[label_idx]]),axis=0)
                new_probabilities = np.expand_dims(new_probabilities,axis=0)

                if mean_probabilities is None:
                        mean_probabilities = new_probabilities
                else:
                        mean_probabilities = np.concatenate([mean_probabilities, new_probabilities])

                features_df = pd.DataFrame(feature_counter.most_common(5),columns=['feature','count'])
                features_df.name = label
                class_soup_list.append(BeautifulSoup(features_df.to_html(),'html.parser'))


        prob_df = pd.DataFrame(mean_probabilities,columns=anecdotes_labels,index=anecdotes_labels)

        #plots
        img_tag = out_soup.new_tag('img',src='label_hist.png',alt='labelhist')
        out_soup.body.insert(0,img_tag)

        html_string_meta = meta_df.to_html()
        meta_soup = BeautifulSoup(html_string_meta,'html.parser')

        html_string_param = param_df.to_html()
        param_soup = BeautifulSoup(html_string_param,'html.parser')

        html_string_main = main_df.to_html()
        main_soup = BeautifulSoup(html_string_main,'html.parser')

        html_string_prob = prob_df.to_html()
        prob_soup = BeautifulSoup(html_string_prob,'html.parser')

        #bottom to top
        for soup in class_soup_list:
                out_soup.body.insert(0,soup)
        out_soup.body.insert(0,prob_soup)
        out_soup.body.insert(0,main_soup)
        out_soup.body.insert(0,param_soup)
        out_soup.body.insert(0,meta_soup)

        #write HTML
        with open("test.html","w") as file:
                file.write(str(out_soup))

        
    
    