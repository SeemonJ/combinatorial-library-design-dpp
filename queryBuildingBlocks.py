from rdkit import Chem
import pandas as pd
import tqdm
import numpy as np

### Sample Sanity check file to verify that the building blocks created contain
### the substructures needed for the reactions
### Amide Coupling - Carboxylic acids and acid chlorides (but this version of LibINVENT did not pass these in reaction filters)
### Buchwald Hartwig - 5 and 6 member aromatic rings with [Cl,I,Br]
### File provided as validation for readLibINVENTOutput RegEx.
smiles = pd.read_csv('example_output/AC_SMILES_split_0.smi', header=None).values
smarts1 = '[#6,#7]1:[#6,#7]:[#6,#7]:[#6,#7]:[#6,#7]:[#6,#7]:1(-[#17,#35,#53])'
smarts2 = '[#6,#7]1:[#6,#7]:[#6,#7]:[#6,#7]:[#6,#7]:1(-[#17,#35,#53])'
smarts3 = '[#6](-[#8;H1])(=[#8])'
smarts4 = '[#6](-[#17])(=[#8])'

smarts_object1 = Chem.MolFromSmarts(smarts1)
smarts_object2 = Chem.MolFromSmarts(smarts2)
smarts_object3 = Chem.MolFromSmarts(smarts3)
smarts_object4 = Chem.MolFromSmarts(smarts4)

ac_list = []
bh_list = []

for i in tqdm.tqdm(smiles):
    try:
        mol = Chem.MolFromSmiles(i)
        atom_num= len(mol.GetAtoms())
        added=False
        if mol.HasSubstructMatch(smarts_object1):
            atom1, atom2= mol.GetSubstructMatch(smarts_object1)[-2:]
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            bond_idx = bond.GetIdx()
            split_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True,  dummyLabels=[(0, 0)])
            fragmented_smiles = Chem.MolToSmiles(split_mol)
            fragment= max(fragmented_smiles.split('.'),key=len)
            bh_list.append(fragment)    
            added = True
        if mol.HasSubstructMatch(smarts_object2):
            atom1, atom2= mol.GetSubstructMatch(smarts_object2)[-2:]
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            bond_idx = bond.GetIdx()
            split_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True,  dummyLabels=[(0, 0)])
            fragmented_smiles = Chem.MolToSmiles(split_mol)
            if added== False:
                fragment= max(fragmented_smiles.split('.'),key=len)
                bh_list.append(fragment)

        added=False
        if mol.HasSubstructMatch(smarts_object3):
            atom1, atom2= mol.GetSubstructMatch(smarts_object3)[:2]
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            bond_idx = bond.GetIdx()
            split_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True,  dummyLabels=[(0, 0)])
            fragmented_smiles = Chem.MolToSmiles(split_mol)
            if added== False:
                fragment= max(fragmented_smiles.split('.'),key=len)
                ac_list.append(fragment)
                added = True
        if mol.HasSubstructMatch(smarts_object4):
            atom1, atom2= mol.GetSubstructMatch(smarts_object4)[:2]
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            bond_idx = bond.GetIdx()
            split_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True,  dummyLabels=[(0, 0)])
            fragmented_smiles = Chem.MolToSmiles(split_mol)
            if added== False:
                fragment= max(fragmented_smiles.split('.'),key=len)
                ac_list.append(fragment)

    except:
        pass

ac_list=list(set(ac_list))
bh_list=list(set(bh_list))    
print("AC BBs: ", len(ac_list))
print("BH BBs: ", len(bh_list))
