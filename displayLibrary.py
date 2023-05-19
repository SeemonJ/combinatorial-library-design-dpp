from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from reinvent_chemistry.library_design.bond_maker import BondMaker
from utils import  get_fingerprints_array, get_fingerprints_list
import pickle

import pickle
with open('publication_example/QSAR_model.pkl','rb') as f:
    clf=pickle.load(f)

def mol_to_fp(mols):
    return get_fingerprints_array(get_fingerprints_list(mols))

def plot_rdkit_svg_grid(mols, mols_per_row=5, filename=None, **kwargs):
    """
    Plots a grid of RDKit molecules in SVG.
    :param mols: a list of RDKit molecules
    :param mols_per_row: size of the grid
    :param filename: save an image with the given filename
    :param kwargs: additional arguments for `RDKit.Chem.Draw.MolsToGridImage`
    :return: the SVG as a string
    """
    core = Chem.MolFromSmiles('N1CCN(CC1)CCCCN')

    img=Draw.MolsToGridImage(mols, molsPerRow=mols_per_row,highlightAtomLists=[mol.GetSubstructMatch(core) for mol in mols],subImgSize=(500, 250), returnPNG=False, **kwargs)

    return img

def join_scaffolds_and_decorations(ac_pos,bh_pos, ac_dict, bh_dict, bondmake):
    tmp=[]
    s='|'
    for i in ac_pos:    
        for j in bh_pos: 
            bh_frag=bh_dict[j]
            ac_frag=ac_dict[i]        
            mol = bondmake.join_scaffolds_and_decorations('[*:0]N1CCN(CC1)CCCCN[*:1]',s.join([bh_frag, ac_frag]))
            tmp.append(mol)    
    return tmp
bondmake=BondMaker()
depth=0

with open(f'example_output/selected_rows.pkl','rb') as f:
    rows=pickle.load(f)
with open(f'example_output/selected_columns.pkl','rb') as f:
    cols=pickle.load(f)
with open("publication_example/AC_Synthon_BB_dict.pkl", "rb") as f:
    ac_dict=pickle.load(f)
with open("publication_example/BH_Synthon_BB_dict.pkl", "rb") as f:
    bh_dict=pickle.load(f)


mols=join_scaffolds_and_decorations(rows[:5],cols[:5], ac_dict, bh_dict, bondmake)
img=plot_rdkit_svg_grid(mols,5,'test_grid')
img.save("example_output/example_selection.png")