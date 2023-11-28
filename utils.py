from typing import Iterable
from rdkit import Chem
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import *
from rdkit.Chem import AllChem
import numpy as np
from scipy.spatial import distance


def calculate_internal_pairwise_similarities(fp_list: np.array) -> np.array:
    """
    Computes the pairwise similarities of the provided list of smiles against itself.

    Returns:
        Symmetric matrix of pairwise similarities. Diagonal is set to zero.
    """
    fps = fp_list
    nfps = len(fps)
    similarities = np.ones((nfps, nfps))
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        
        similarities[i, :i] = sims
        similarities[:i, i] = sims

    return similarities

def calculate_internal_pairwise_similarities_array(fp_list: np.array) -> np.array:

    return distance.squareform(distance.pdist(fp_list,'jaccard'))

def get_fingerprints_list(mols: Iterable[Chem.Mol], radius=3, length=2048):

    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols]

def get_fingerprints_array(fp_list, size = 2048):

   
    nfp = np.zeros((len(fp_list), size), np.float32)
    for i, fp in enumerate(fp_list):
        for idx, v in enumerate(fp): 
            nfp[i, idx] = v
    return nfp

def get_mols(smiles_list: Iterable[str]) -> Iterable[Chem.Mol]:
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
        except:
            continue
        if mol is not None:
            yield mol