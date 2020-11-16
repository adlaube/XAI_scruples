from anecdotes_LIME import explain_anecdote_lime
from anecdotes_utils import anecdotes_data,anecdotes_labels

import numpy as np
import pandas as pd

if __name__ == "__main__":

        #np.random.randint(0,2500,50,np.uint32)
        sample_indices = np.array([1491, 2291, 1636,  146,  204, 2092, 1138,  319,  645,  315, 2243,
                                 2357, 1616, 1853,  671,  762, 1240, 1675,  944,  220, 2304,  567,
                                         434,  946,  220, 1317,  877, 2077, 1559, 1509,  439,  670, 2470,
                                         203, 1949,  138, 2437,  506,  401, 2212, 1652,  807,  221,  110,
                                 2478,  567,  551,   69, 1911, 1851],dtype=np.uint32)
        
        anecdotes_df = pd.DataFrame(anecdotes_data)
        anecdotes_df = anecdotes_df.iloc[sample_indices]

        sample_feature_len_hist = anecdotes_df['text'].str.len()
        sample_label_hist = anecdotes_df['label'].value_counts()
        sample_type_hist = anecdotes_df['post_type'].value_counts()
        #label_scores count useful?

        param_dict = {"max_number_of_pertubations"      :1,
                        "adaptive_pertubations"         :False,
                        "number_of_features"            :10,
                        }



        for idx in sample_indices:
                exp = explain_anecdote_lime(idx,param_dict)
                
                #explain_anecdote_lime(idx,10,10000)
        
    
    