import pandas as pd
import pickle
import os
import tqdm


ac_tree_load = pd.read_csv('publication_example/AC_full.csv')
bh_tree_load = pd.read_csv('publication_example/BH_full.csv')

for depth in range(1):
    for k in tqdm.tqdm(range(2)):
        ac_tree = ac_tree_load.loc[[i<=depth for i in ac_tree_load['depths'].values]]
        bh_tree = bh_tree_load.loc[[i<=depth for i in bh_tree_load['depths'].values]]
        best_rows = ac_tree.sample(n=12)
        best_columns = bh_tree.sample(n=8)
        
        best_rows=list(best_rows['Products'].values)
        best_columns=list(best_columns['Products'].values)

        prefix = f'Results_random_{depth}_{k}'
        os.makedirs(prefix,exist_ok=True)

        with open(f'{prefix}/selected_rows.pkl','wb') as f:
            pickle.dump(best_rows,f)    
        with open(f'{prefix}/selected_columns.pkl','wb') as f:
            pickle.dump(best_columns,f)