"""
    A Unified Perspective on Regularization and Perturbation in Differentiable Subset Selection
    
    function: output the evaluation table.
    
"""
import json,time
import numpy as np
import pandas as pd
from IPython.display import display
from os.path import join
# save data to json file
if __name__=="__main__":
    K = 50
    figure_dir = 'figures'

    # name = '%d_perturbed_debug_mnist_%d.json' % (i, K)
    # name = '%d_regularized_concrete_mnist_%d.json' % (i, K)
    table = np.zeros((10, 6, 3))
    for t in range(10):
        name = '%d_concrete_regularized_perturbed_mnist_%d.json' % (t, K)
    
        
        with open(join(figure_dir, name), 'r') as f:
            data = json.load(f)
            print(data)
        
        criterias = ['NMI', 'ACC', 'MSELR', 'MSE', 'ClASS', 'CLASSDT']
        # methods = ['pca_extractor', 'autoencoder',  'lap_ours', 'AEFS', 'concrete_column_subset_selector_general', 'udfs_ours', 'mcfs_ours',    'pfa_transform',  'random_transform']
        methods = [ 'concrete_column_subset_selector_general','perturbed_column_subset_selector_general', 'regularized_column_subset_selector_general', ]

        print(data.keys())
        
        
        index = '%d' % (K)
        print(index)
        
        j=0
        for key1 in data.keys():
            i=0
            print(key1)
            for key2 in data[key1]:
                print(key2)
                table[t, i, j] = '%.3f' % data[key1][key2][index]
                i +=1
            j +=1
    print(table)
    
    table = np.mean(table, axis = 0)
    df = pd.DataFrame()
    df['criteria'] = criterias
    
    df[methods] = table
    path = 'table'
    # df.to_csv(join(path,r"perturbed_result_plot_{}.csv".format(K)), index=False, sep=',')
    df.to_csv(join(path,r"reg_concrete_result_plot_{}.csv".format(K)), index=False, sep=',')
    display(df)
