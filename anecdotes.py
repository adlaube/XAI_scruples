from anecdotes_LIME import explain_anecdote_lime

import numpy as np



if __name__ == "__main__":

        #np.random.randint(0,2500,50,np.uint32)
        test_indices = np.array([1491, 2291, 1636,  146,  204, 2092, 1138,  319,  645,  315, 2243,
                                2357, 1616, 1853,  671,  762, 1240, 1675,  944,  220, 2304,  567,
                                        434,  946,  220, 1317,  877, 2077, 1559, 1509,  439,  670, 2470,
                                        203, 1949,  138, 2437,  506,  401, 2212, 1652,  807,  221,  110,
                                2478,  567,  551,   69, 1911, 1851],dtype=np.uint32)
        

        for idx in test_indices:
                explain_anecdote_lime(idx,10)

        
    
    