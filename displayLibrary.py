from rdkit import Chem
from rdkit.Chem import Draw
from reinvent_chemistry.library_design.bond_maker import BondMaker
from utils import  get_fingerprints_array, get_fingerprints_list
import pickle
import argparse

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

def join_scaffolds_and_decorations(ac_pos,bh_pos, ac_dict, bh_dict, bondmake,scaffold):
    tmp=[]
    s='|'
    for i in ac_pos:    
        for j in bh_pos: 
            bh_frag=bh_dict[j]
            ac_frag=ac_dict[i]        
            mol = bondmake.join_scaffolds_and_decorations(scaffold,s.join([bh_frag, ac_frag]))
            tmp.append(mol)    
    return tmp

def libPlotter(input_a,input_b,dict_a,dict_b,output,scaffold):
    with open(input_a,'rb') as f:
        rows=pickle.load(f)
    with open(input_b,'rb') as f:
        cols=pickle.load(f)
    with open(dict_a, "rb") as f:
        ac_dict=pickle.load(f)
    with open(dict_b, "rb") as f:
        bh_dict=pickle.load(f)

    bondmake=BondMaker()

    mols=join_scaffolds_and_decorations(rows[:5],cols[:5], ac_dict, bh_dict, bondmake,scaffold)
    img=plot_rdkit_svg_grid(mols,5,'test_grid')
    img.save(output)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
                                     'Plotting script from library optimization')

    parser.add_argument("--input_a", "-ia", help='File containting all building blocks of type A',\
        type=str, required=True)
    parser.add_argument("--input_b", "-ib", help='File containting all building blocks of type B',\
        type=str, required=True)
    parser.add_argument("--dict_file_a", "-da", help='Dict File containting mapping for all building blocks of type A to synthon',\
        type=str, required=True)
    parser.add_argument("--dict_file_b", "-db", help='Dict File containting mapping for all building blocks of type B to synthon',\
        type=str, required=True)
    parser.add_argument("--output", "-o", help='output directory of optimization',\
        type=str, required=True)
    parser.add_argument("--scaffold", "-sc", help='SMILES string of the scaffold to attach the BBs to',\
        default= '[*:0]N1CCN(CC1)CCCCN[*:1]', type=str, required=False)
    
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    runner = libPlotter(args['input_a'],args['input_b'],args['dict_file_a'],args['dict_file_b'],args['output'],args['scaffold'])
            