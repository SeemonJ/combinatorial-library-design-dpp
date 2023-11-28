from utils import calculate_internal_pairwise_similarities_array, get_fingerprints_array, get_mols, get_fingerprints_list
import numpy as np
from numpy.random import default_rng
from reinvent_chemistry.library_design.bond_maker import BondMaker
from rdkit import Chem
import tqdm


class library():
    def __init__(self, seed, import_row_len, import_column_len, n_rows, n_cols, ac_dict, bh_dict, ac_df, bh_df, clf, scaffold):
        self.seed = default_rng(seed)
        self.clf = clf
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.size=n_rows*n_cols
        self.dimensions=[import_row_len, import_column_len]
        self.bondmake=BondMaker()
        self.ac_dict = ac_dict
        self.bh_dict=bh_dict
        self.ac_df=ac_df
        self.bh_df=bh_df
        self.ac_smiles = ac_df['Products'].values
        self.ac_fps = self.mol_to_fp(get_mols(self.ac_smiles))
        self.bh_smiles = bh_df['Products'].values
        self.bh_fps = self.mol_to_fp(get_mols(self.bh_smiles))
        self.scaffold=scaffold

        
    def predict(self,input):
        return self.clf.predict_proba(input)

    def getLogDet(self,fps):
        currentMatrix =  1- calculate_internal_pairwise_similarities_array(fps).reshape(len(fps), len(fps))
        _, currentDet =  np.linalg.slogdet(currentMatrix)

        return currentDet

    def mol_to_fp(self,mols):
        return get_fingerprints_array(get_fingerprints_list(mols))
        
    def join_scaffolds_and_decorations(self,ac_smiles,bh_smiles):
        tmp=[]
        s='|'
        for i in ac_smiles:
            for j in bh_smiles:
                bh_frag=self.bh_dict[j]
                ac_frag=self.ac_dict[i]
                mol = self.bondmake.join_scaffolds_and_decorations(self.scaffold,s.join([bh_frag, ac_frag]))
                tmp.append(mol)    

        
        return tmp

    def get_InitialSolution(self, weights):
        
        # Producing a random configuration
        initial_rows = self.seed.choice(self.dimensions[0],self.n_rows,replace = False)
        initial_columns = self.seed.choice(self.dimensions[1],self.n_cols,replace = False)
        row_smiles = self.ac_smiles[initial_rows]
        col_smiles = self.bh_smiles[initial_columns]
        current_objval=1
        current_selection= self.join_scaffolds_and_decorations(row_smiles,col_smiles)
        # Calculating implemented properties
        self.score_dict={}
        self.current_props={}
        self.weights = weights
        self.current_props['FPs']  = self.mol_to_fp(current_selection)
        # Storing 
        if self.weights[0]>0:
            current_QED_list = [] 
            for i in current_selection:
                current_QED_list.append(Chem.QED.qed(i))
            self.current_props['QED'] =np.array(current_QED_list)
            current_qed_score = sum(self.current_props['QED'])/self.size
            self.score_dict['QED']=current_qed_score
            current_objval=current_objval*current_qed_score            
        if self.weights[1]>0:
            self.current_props['QSAR'] = np.array(self.predict(self.mol_to_fp(current_selection)))[:,1]
            current_qsar_score =  np.mean(self.current_props['QSAR'])
            self.score_dict['QSAR']=current_qsar_score
            current_objval=current_objval*current_qsar_score
        if self.weights[2]>0:
            current_div = self.getLogDet(self.current_props['FPs'])
            self.score_dict['Diversity']=current_div
            current_objval=current_objval*current_div

        return [initial_rows, initial_columns], current_objval

    def Objfun(self, new_idx, current, pick, phase):
        '''
        Return the objective function value of the solution
        new_idx: The index of the building block to replace
        current: array containing the indicies for current building blocks
        pick: which dimensions (row or column) to change
        '''
        rows=current[0]
        columns=current[1]
        ac_smiles=self.ac_smiles[rows]
        bh_smiles=self.bh_smiles[columns]
        ignore_flag=False
        candidate_score = 1
        k = self.seed.choice(len(current[pick]))
        candidate_update_scores={}
        candidate_update_lists={}
        frag_list=[]
        if pick ==0:
            new_smiles=[self.ac_smiles[new_idx[0]]]
            new_products =self.join_scaffolds_and_decorations(new_smiles,bh_smiles)
            swap_idxs=range(k*self.n_cols,(k+1)*self.n_cols)
            for l in ac_smiles:
                frag_list.append(self.ac_dict[l])
            new_fragment=self.ac_dict[new_smiles[0]]
            if new_fragment in frag_list:
                # Depending on how you setup your list of building blocks
                # during AiZynthFinder you can have two building blocks 
                # that create the same moiety on the product, so we check 
                # for this and ignore the configuration if it did.
                print("Identical building blocks picked, check input data")
                print(frag_list)
                ignore_flag = True
                
        if pick ==1:
            new_smiles=[self.bh_smiles[new_idx[0]]]
            new_products =self.join_scaffolds_and_decorations(ac_smiles,new_smiles)
            swap_idxs=list(range(k,self.size,self.n_cols))
            for l in bh_smiles:
                frag_list.append(self.bh_dict[l])
            new_fragment=self.bh_dict[new_smiles[0]]
            if new_fragment in frag_list:
                print("Identical building blocks picked, check input data")
                print(frag_list)
                ignore_flag = True
                
        if ignore_flag == False:
            if self.weights[0]>0:
                candidate_qed = np.copy(self.current_props['QED'])
                qed = []
                for l in new_products:
                    qed.append(Chem.QED.qed(l))
                new_qed = np.array(qed)
                candidate_qed[swap_idxs] = new_qed
                qed_score = (sum(candidate_qed)/(self.size))
                update= np.power((qed_score/self.score_dict['QED']),self.weights[0])
                candidate_update_scores['QED']=qed_score
                candidate_update_lists['QED']=candidate_qed
                candidate_score = candidate_score*update
            
            if self.weights[1]>0:
                candidate_qsar = np.copy(self.current_props['QSAR'])
                new_qsar=np.array(self.predict(self.mol_to_fp(new_products)))[:,1]
                candidate_qsar[swap_idxs] = new_qsar
                qsar_score= (sum(candidate_qsar)/(self.size))
                update =  np.power((qsar_score/self.score_dict['QSAR']),self.weights[1])
                candidate_update_scores['QSAR']=qsar_score
                candidate_update_lists['QSAR']=candidate_qsar
                candidate_score = candidate_score*update

            if self.weights[2]>0 :
                candidate_fps = np.copy(self.current_props['FPs'])
                new_fps=self.mol_to_fp(new_products)
                candidate_fps[swap_idxs] = new_fps
                div_score =self.getLogDet(candidate_fps)
                update = np.power(np.exp(div_score-self.score_dict['Diversity']),self.weights[2])
                candidate_update_scores['Diversity']=div_score
                candidate_update_lists['FPs']=candidate_fps
                candidate_score = candidate_score*update
            
        swap_idx=k
        
        
        return  candidate_score, candidate_update_scores, candidate_update_lists,  swap_idx, ignore_flag 


    def SwapMove(self, vector, i ,j):
        '''Takes a list (solution)
        returns a new neighbor solution with i, j swapped
       '''
        #Swap
        candidate = vector.copy()
        candidate[i] = j
        
        return candidate

    def multi_obj(self, tol, weights = [2,2,0.015]):
      
  
        current, candidate_objvalue = self.get_InitialSolution(weights)
        best_rows = current[0]
        best_columns = current[1]
        since_last = 0
        n=0
        phase=1
        while since_last < tol:
            print('\n\n### iter {} ###  Acceptance Ratio: {}, Time since last improvement: {}'
                  .format(n, candidate_objvalue, since_last), flush=True)
            for key in self.score_dict.keys():
                print(f"{key} score: {self.score_dict[key]}")
            since_last+=1
            ignore_flag= False
            # Choose dimension
            pick=self.seed.choice(2,1)[0]
            

            new_idx=self.seed.choice(self.dimensions[pick],1)
            while np.in1d(current[pick], new_idx, assume_unique=True).any():
                # For larger selection spaces it is faster to draw a pick until you
                # get a new index, rather than enumerate the possible choices. 
                new_idx=self.seed.choice(self.dimensions[pick],1)
    

            candidate_objvalue, candidate_update, candidate_list, worst_idx,ignore_flag  = self.Objfun(new_idx, current, pick, phase)
            candidate_solution = self.SwapMove(current[pick], worst_idx, new_idx)   
            if not ignore_flag :
                if (candidate_objvalue>1):
                    current[pick]=candidate_solution
                    for key in self.score_dict.keys():
                        self.score_dict[key]=candidate_update[key]
                    for key in self.current_props.keys():
                        self.current_props[key]= candidate_list[key]
             
                    best_rows=self.ac_smiles[current[0]]
                    best_columns=self.bh_smiles[current[1]]
                    since_last=0
            n+=1

        return best_rows, best_columns
    