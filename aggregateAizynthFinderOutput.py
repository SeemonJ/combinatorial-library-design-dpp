import pandas as pd
from aizynthfinder.reactiontree import ReactionTree
from tqdm import tqdm
import glob
import re
import pickle

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

with open(f"publication_example/AC_Synthon_BB_dict.pkl", "rb") as f:
    BB_dict=pickle.load(f)

def extract_from_tree(big_frame):
    dict_list=[]
    added_dict={}
    bb_idxs={}
  
    for idx,product in tqdm(enumerate(big_frame.trees.values)):
        prod_smiles=product[00]['smiles']
        
        depths=[]
 
        for itree, tree in enumerate(product):
            treeobj=ReactionTree.from_dict(tree)
            
            if treeobj.is_solved:
                react=treeobj.leafs()
                candidate_depth = []
                for i in react:
                    depth=treeobj.depth(i)/2
                    if depth is not None:  
                        candidate_depth.append(depth)
                depths.append(max(candidate_depth))
            else:
                depths.append(11)
        if BB_dict[prod_smiles] in added_dict:
            if added_dict[BB_dict[prod_smiles]]>min(depths):
                added_dict[BB_dict[prod_smiles]]=min(depths)
                bb_idxs[BB_dict[prod_smiles]]=idx
        else:
            added_dict[BB_dict[prod_smiles]]=min(depths)
            bb_idxs[BB_dict[prod_smiles]]=idx
        dict_list.append({'Products' : prod_smiles, 'depths' : min(depths)})
    return dict_list, bb_idxs

# get data file names
filenames = sorted_alphanumeric(glob.glob(f"publication_example/*.hdf5"))


dfs = []
for filename in filenames:
    dfs.append(pd.read_hdf(filename, "/table"))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)

output_dict, to_keep = extract_from_tree(big_frame)
res = list(to_keep.values())
df_new = pd.DataFrame(columns=['Products', 'depths'])
df_new = df_new.append(output_dict, ignore_index=True)
df_new=df_new.loc[res]
df_new.to_csv(f'AC_example.csv')



