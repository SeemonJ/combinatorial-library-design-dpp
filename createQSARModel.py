import numpy as np
import pandas as pd
from utils import get_fingerprints_array,get_mols, get_fingerprints_list
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier


def create_complimentary_prob(probs):
    return np.c_[np.ones(len(probs))*(1-probs), probs]
    
def create_activity_labels(activities, threshold):
    return np.c_[np.ones(len(activities))-activities>threshold, activities > 6]

    
def createQSAR(filename,threshold):
    df = pd.read_csv(filename).drop_duplicates(subset=['SMILES']).dropna()
    oob_scores=[]
    test_scores=[]
    
    train_slice,  test_slice = \
            np.split(df.sample(frac=1), 
                    [int(.8*len(df))])
    smiles_train = train_slice['SMILES']
    smiles_test = test_slice['SMILES']
    train_activity=train_slice['pXC50'].reset_index()
    test_activity=test_slice['pXC50'].reset_index()
    train_activity=create_activity_labels(train_activity['pXC50'].values, threshold)
    test_activity=create_activity_labels(test_activity['pXC50'].values, threshold)
    # You might need to sanitize the input file from RDKit unparseable SMILES
    train_mols= get_mols(smiles_train)
    test_mols= get_mols(smiles_test)
    train_fp=get_fingerprints_array(get_fingerprints_list(train_mols))
    test_fp = get_fingerprints_array(get_fingerprints_list(test_mols))
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,oob_score=True)
    clf.fit(train_fp,train_activity[:,1])        
    p=clf.score(test_fp,test_activity[:,1])
    test_scores.append(p)
    oob_scores.append(clf.oob_score_)
    print("Oob score:", clf.oob_score_)
    print("Test Auroc:",p)

    with open('exampleQSAR.pkl','wb') as f:
        pickle.dump(clf,f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=
                                     'Creating QSAR model from ExcapeDB .csv')
    parser.add_argument("--input", "-i", help='File containting SMILES and pXC50 data',\
         type=str, required=False)
    parser.add_argument("--threshold", "-t", help='pXC50 threshold for active/inactive',\
        default=1, type=int, required=False)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    
    createQSAR(args['input'], args['threshold'])
