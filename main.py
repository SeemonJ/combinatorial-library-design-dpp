import pandas as pd
import numpy as np
import pickle
from optimizeLibrary import library
import json
import argparse
import os
import matplotlib.pyplot as plt

    
class libDecisionMaker():
    def __init__(self, rows, columns, TreeDepth=0, rng_seed=123):
        self.depth = TreeDepth
        self.rows = rows
        self.rng_seed = rng_seed
        self.columns = columns

        
    
    def run(self, ac_file, bh_file, scaffold,index=0, steps = 10000):
        swap_size=2
        ac_tree_load = pd.read_csv(ac_file)
        bh_tree_load = pd.read_csv(bh_file)
    
        ac_tree = ac_tree_load.loc[[i<=self.depth for i in ac_tree_load['depths'].values]]
        bh_tree = bh_tree_load.loc[[i<=self.depth for i in bh_tree_load['depths'].values]]
        
        n_ac = ac_tree.shape[0]
        n_bh = bh_tree.shape[0]
        print("Optimizing over {} rows and {} columns".format(n_ac,n_bh), flush=True)
        
        with open("publication_example/AC_Synthon_BB_dict.pkl", "rb") as f:
            ac_dict=pickle.load(f)
        with open("publication_example/BH_Synthon_BB_dict.pkl", "rb") as f:
            bh_dict=pickle.load(f)
        
        with open('publication_example/QSAR_model.pkl','rb') as f:
            clf=pickle.load(f)

        multi_ = library(self.rng_seed, n_ac, n_bh, self.rows, self.columns, ac_dict, bh_dict, ac_tree, bh_tree, clf, scaffold)
        best_rows, best_columns= multi_.multi_obj(steps, swap_size=swap_size)
        print("Run finished, with rows: {} and columns {}".format(best_rows, best_columns))
        prefix = f'example_output'
        os.makedirs(prefix,exist_ok=True)
        with open(f'{prefix}/selected_rows.pkl','wb') as f:
            pickle.dump(best_rows,f)    
        with open(f'{prefix}/selected_columns.pkl','wb') as f:
            pickle.dump(best_columns,f)
        
  
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
                                     'Selection algorithm for LibINVENT')
    parser.add_argument("--rows", "-r", help='Number of rows to select',\
        default=12, type=int, required=False)
    parser.add_argument("--columns", "-c", help='Number of columns to select',\
        default=8, type=int, required=False)
    parser.add_argument("--steps", "-e", help='Number of Steps per epoch',\
        default=10, type=int, required=False)
    parser.add_argument("--depth", "-d", help='reaction tree depth in molecule dataset',\
        default=0, type=int, required=False)
    parser.add_argument("--index", "-i", help='Index for batch runs (also used to vary seed per run)',\
        default=1, type=int, required=False)
    parser.add_argument("--seed", "-se", help='Seed for batch runs (multiplied with suffix)',\
        default=1414, type=int, required=False)
    parser.add_argument("--block_file_a", "-ba", help='File containting all building blocks of type A',\
        default='publication_example/AC_full.csv', type=str, required=False)
    parser.add_argument("--block_file_b", "-bb", help='File containting all building blocks of type B',\
        default='publication_example/BH_full.csv', type=str, required=False)
    parser.add_argument("--scaffold", "-sc", help='SMILES string of the scaffold to attach the BBs to',\
        default= '[*:0]N1CCN(CC1)CCCCN[*:1]', type=str, required=False)
    
    
 
  

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    runner = libDecisionMaker(args['rows'],args['columns'],TreeDepth=args['depth'],rng_seed=args['index']*args['seed'])
    runner.run(steps=args['steps'], ac_file=args['block_file_a'],bh_file=args['block_file_b'], index=args['index'], scaffold = args['scaffold'])
            
            
        