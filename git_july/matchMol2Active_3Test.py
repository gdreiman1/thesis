# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:54:59 2019

@author: gdrei
"""
'''Creating files that have mol and activity strings attached to eachother
that way we can oversample upstream of the pytorch implemention if needed'''


import glob
import os
import pandas as pd
from rdkit import Chem
import pickle

def Consistant_Checker(expr_table):
    #remove nans

    nan_inds = expr_table.index[expr_table['PUBCHEM_CID'].isna()].tolist()
    expr_table.drop(nan_inds,inplace = True)
    #find duplicated cids, removed those that have inconsistent results, return those removed cids
    from collections import Counter
    duplicated_cid = [k for k,v in Counter(expr_table.loc[:,'PUBCHEM_CID']).items() if v>1]
    dropped = []
    for x in duplicated_cid:
        #get locations of duplicated CID
        dup_indxs = expr_table.index[expr_table['PUBCHEM_CID']==x].tolist()
        #boolean where results beeing all the same returns 1
        if len(set(expr_table.loc[dup_indxs, 'PUBCHEM_ACTIVITY_OUTCOME'])):
            #drop repated experiemnts after first instance
            expr_table.drop(dup_indxs[1:],inplace = True)
        else:
            #its inconsistent, drop everything
            expr_table.drop(dup_indxs,inplace = True)
            dropped.append(x)
    expr_table['PUBCHEM_CID'] =expr_table['PUBCHEM_CID'].astype(int).astype(str)
    return(expr_table,dropped)
def process_and_save(AID):
    AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
    path_to_sdf = glob.glob(AID_path+'/*.sdf')[0]
    expr_loc = glob.glob(AID_path+'/*.csv')[0]
    save_path = AID_path+ '/' + AID +'mol_processed_partest.pkl'
    # this deals out SD formatted molecules
    suppl = Chem.SDMolSupplier(path_to_sdf)
    #get list of dicts and turn it into a df
    prop_dict_list = [{'PUBCHEM_COMPOUND_CID':mol.GetProp('PUBCHEM_COMPOUND_CID'),'MOL':mol} for mol in suppl]
    #turn into df
    CID_Smiles_df = pd.DataFrame(prop_dict_list)
    #get active info as a dp
    expr_table = pd.read_csv(expr_loc)
    #remove duplicated CIDs
    expr_table,_ = Consistant_Checker(expr_table)
    #merge activty data with smiles
    final_table = expr_table[['PUBCHEM_CID','PUBCHEM_ACTIVITY_OUTCOME']].merge(CID_Smiles_df, left_on='PUBCHEM_CID', right_on='PUBCHEM_COMPOUND_CID')
    final_table.drop('PUBCHEM_COMPOUND_CID',axis = 1,inplace = True)
    pickle_on = open(save_path,'wb')
    pickle.dump(final_table,pickle_on)
    pickle_on.close()
#for AID in ['AID_628','AID_894','AID_449739','AID_624255','AID_1345083']:
#for AID in ['AID_893']:  

for AID in ['AID_1259354', 'AID_598', 'AID_488969']:
    AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
    path_to_sdf = glob.glob(AID_path+'/*.sdf')[0]
    expr_loc = glob.glob(AID_path+'/*.csv')[0]
    save_path = AID_path+ '/' + AID +'mol_processed.pkl'
    # this deals out SD formatted molecules
    suppl = Chem.SDMolSupplier(path_to_sdf)
    #get list of dicts and turn it into a df
    prop_dict_list = [{'PUBCHEM_COMPOUND_CID':mol.GetProp('PUBCHEM_COMPOUND_CID'),'MOL':mol} for mol in suppl]
    #turn into df
    CID_Smiles_df = pd.DataFrame(prop_dict_list)
    #get active info as a dp
    expr_table = pd.read_csv(expr_loc)
    #remove duplicated CIDs
    expr_table,_ = Consistant_Checker(expr_table)
    #merge activty data with smiles
    final_table = expr_table[['PUBCHEM_CID','PUBCHEM_ACTIVITY_OUTCOME']].merge(CID_Smiles_df, left_on='PUBCHEM_CID', right_on='PUBCHEM_COMPOUND_CID')
    final_table.drop('PUBCHEM_COMPOUND_CID',axis = 1,inplace = True)
    pickle_on = open(save_path,'wb')
    pickle.dump(final_table,pickle_on)
    pickle_on.close()

from joblib import Parallel,delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
AID_list = ['AID_1259354', 'AID_598', 'AID_488969']
Parallel(n_jobs=num_cores, verbose=50)(delayed(process_and_save)(i)for i in AID_list)
