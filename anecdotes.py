from anecdotes_LIME import explain_anecdote_lime
from anecdotes_utils import anecdotes_data,anecdotes_labels

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

if __name__ == "__main__":

        #np.random.randint(0,2500,50,np.uint32)
        sample_indices = np.array([1491, 2291, 1636,  146,  204, 2092, 1138,  319,  645,  315, 2243,
                                 2357, 1616, 1853,  671,  762, 1240, 1675,  944,  220, 2304,  567,
                                         434,  946,  220, 1317,  877, 2077, 1559, 1509,  439,  670, 2470,
                                         203, 1949,  138, 2437,  506,  401, 2212, 1652,  807,  221,  110,
                                 2478,  567,  551,   69, 1911, 1851],dtype=np.uint32)

        sample_indices = np.array([1,2,3])
        sample_indices = np.sort(sample_indices)
        
        ## MAIN DATAFRAME
        anecdotes_df = pd.DataFrame(anecdotes_data)
        anecdotes_df = anecdotes_df.iloc[sample_indices]

        sample_feature_len_hist = anecdotes_df['text'].str.len()
        sample_label_hist = anecdotes_df['label'].value_counts()
        sample_type_hist = anecdotes_df['post_type'].value_counts()
        #label_scores count useful?

        #hist with feature scores

        param_dict = {"max_number_of_pertubations"      :10,
                        "adaptive_pertubations"         :False,
                        "number_of_features"            :10,
                        }     

        out_soup = None

        #init dataframes
        columns_df = ["features","contributions"]
        explanation_dataframes_dict = {}
        exp_lists = []
        for label in anecdotes_labels:
                exp_lists.append([]) 

        for idx in sample_indices:

                exp = explain_anecdote_lime(idx,param_dict)
                # List processing
                for label_idx in range(len(anecdotes_labels)):
                        exp_of_one_class_tuple = exp.as_list(label_idx)
                        exp_lists[label_idx] += (exp_of_one_class_tuple)
                
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

        #compile dataframes
        for label_idx in range(len(anecdotes_labels)):
                exp_df = pd.DataFrame(exp_lists[label_idx],columns=columns_df)
                label = anecdotes_labels[label_idx]
                explanation_dataframes_dict[label] = exp_df



        #write HTML
        with open("test.html","w") as file:
                file.write(str(out_soup))

        
    
    