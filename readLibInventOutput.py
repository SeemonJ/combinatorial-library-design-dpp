import pandas as pd
from rdkit import Chem
import tqdm
import pickle
import argparse
import os
import numpy as np

class libinventReader():
    def __init__(self, input,output_dir, threshold,num_splits):
        self.input=input
        self.output_dir=output_dir
        self.threshold=threshold
        self.num_splits=num_splits
        
    def run(self):
        data = pd.read_csv(self.input)
        wrong_reactions = data[data['reaction_filters'] < 1.0].index
        print("Number Products Total:", len(data))
        data.drop(wrong_reactions, inplace=True)
        print("Number Products matching reactions :", len(data))
        low_scores = data[data['DRD2'] <= self.threshold].index
        data.drop(low_scores, inplace=True)
        print(f"Number Products with DRD2 above {self.threshold}:", len(data))
        scaffold= data.Scaffold.values
        sample_decorations = [x.split('|') for x in scaffold]
        AC_compound = []
        BH_compound = []
        AC_dict = {}
        BH_dict = {}

        for k, in tqdm.tqdm(zip(sample_decorations)):
            ### Generate Synthon to Full Molecule for input to retrosynthesis
            ### Since we know for this study that we use Primary Amide Coupling and 
            ### Buchwald-Hartwig, we know what the generated synthons from LibINVENT must
            ### be without needing ChemInformatics knowledge and can conduct the transformation
            ### via RegEX of the output.
            BH_addition = Chem.MolToSmiles(Chem.MolFromSmiles(k[1].replace("*","[Cl]")))
            BH_addition_2 = Chem.MolToSmiles(Chem.MolFromSmiles(k[1].replace("*","Br")))
            BH_addition_3 = Chem.MolToSmiles(Chem.MolFromSmiles(k[1].replace("*","I")))
            AC_addition = Chem.MolToSmiles(Chem.MolFromSmiles(k[2].replace("*","O")))
            BH_compound.append(BH_addition)
            BH_compound.append(BH_addition_2)
            BH_compound.append(BH_addition_3)
            AC_compound.append(AC_addition)
            ### Create Dictionary with canonicalized SMILES -> Synthon used
            BH_dict[Chem.MolToSmiles(Chem.MolFromSmiles(BH_addition))] = k[1]
            BH_dict[Chem.MolToSmiles(Chem.MolFromSmiles(BH_addition_2))] = k[1]
            BH_dict[Chem.MolToSmiles(Chem.MolFromSmiles(BH_addition_3))] = k[1]
            AC_dict[Chem.MolToSmiles(Chem.MolFromSmiles(AC_addition))] = k[2]

        AC_set = set(AC_compound)
        BH_set = set(BH_compound)
        AC_compound_df = pd.DataFrame({"SMILES": list(AC_set)})
        BH_compound_df = pd.DataFrame({"SMILES": list(BH_set)})
        prefix = self.output_dir
        os.makedirs(prefix,exist_ok=True)
        with open(f"{prefix}/AC_Synthon_BB_dict.pkl", "wb") as f:
            pickle.dump(AC_dict, f)
        with open(f"{prefix}/BH_Synthon_BB_dict.pkl", "wb") as f:
            pickle.dump(BH_dict, f)    
        
        ### Split Building blocks to different files for batch-processing in AiZynthfinder
        ac_splits = np.array_split(AC_compound_df,self.num_splits)
        bh_splits = np.array_split(BH_compound_df,self.num_splits)
        for i,j in enumerate(ac_splits):
            j.to_csv(f"{prefix}/AC_SMILES_split_{i}.smi", header=False, index=False)
        for i,j in enumerate(bh_splits):
            j.to_csv(f"{prefix}/BH_SMILES_split_{i}.smi", header=False, index=False)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
                                     'Post Processing of LibINVENT output')

    parser.add_argument("--output", "-o", help='output directory of processing',\
        default="sample_run", type=str, required=False)
    parser.add_argument("--input", "-i", help='LibINVENT log to be processed',\
        default="publication_example/LibINVENT_output.csv", type=str, required=False)
    parser.add_argument("--num_splits", "-n", help='Number of split batches of smiles (for AiZynthfinder processing)',\
        default=128, type=int, required=False)
    parser.add_argument("--threshold", "-t", help='QSAR threshold for keeping BBs (for AiZynthfinder processing)',\
        default=0.8, type=float, required=False)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    
    reader=libinventReader(args['input'],args['output'],args['threshold'],args['num_splits'])
    reader.run()
