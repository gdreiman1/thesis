# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:52:31 2019

@author: gdrei

This script takes raw data tables and CID based SDF files from pubchem and 
generates 1024 bit Morgan Finger Print Hashes concatenated with a 97x1 chemical/
physical property feature vector.
"""

#Imports

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import matplotlib.pyplot as plt
import pickle
from comet_ml import Experiment

#%%
'''Read in our results'''
expr_table = pd.read_csv(r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_1345083_datatable.csv')
path_to_file = r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\1668495245601928809.sdf\1668495245601928809.sdf'
# this deals out SD formatted molecules
suppl = Chem.SDMolSupplier(path_to_file)
#make a df to hold fingerprint, activity, and ID
activity = pd.DataFrame(expr_table.loc[5:,'PUBCHEM_ACTIVITY_OUTCOME'], index=expr_table.loc[5:,'PUBCHEM_CID'])
#fp_activity.join(expr_table.loc[:,'PUBCHEM_ACTIVITY_OUTCOME'])
#%%
# check consistancy
def Consistant_Checker(expr_table):
    #remove nans

    nan_inds = expr_table.index[expr_table['PUBCHEM_CID'].isna()].tolist()
    expr_table.drop(nan_inds,inplace = True)
    #find duplicated cids, removed those that have inconsistent results, return thoe removed cids
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

def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))
#get morgen fps:
def get_mfp_hash(fp_length,suppl):
    fp_list = [[mol.GetProp('_Name'),ExplicitBitVect_to_NumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits = fp_length))] for mol in suppl]
    return fp_list
#align the activiy with the fps
def Align_with_Activity(activity_table,char_list, removed):
    '''status = 1 for fp, status = 0 for mol_char'''
    srt = {b: i for i, b in enumerate(expr_table.loc[:,'PUBCHEM_CID'])}
    
    #remove cids from char_list if we removed them before
    #go thru outer list, getting char rows. If the cid in char_row[0] is not in removed (list of cids that were removed)
    char_list[:] = [char_row for char_row in char_list if char_row[0] not in removed]

            
    #when same length its mfps so creat header 
    if len(activity_table) == len(char_list):
        name_vect = ['name','MFP']
        to_sort = char_list
        sorted_chars = np.array(sorted(to_sort, key=lambda x: srt[x[0]]))
        activity_table['MFP'] = sorted_chars[:,1]
    else:
        name_vect = char_list[0]
        to_sort = char_list[1:]
        sorted_chars = np.array(sorted(to_sort, key=lambda x: srt[x[0]]))
        for i,char_name in enumerate(name_vect[1:],1):
            activity_table[char_name] = sorted_chars[:,i]
    return(activity_table)
def Add_Mol_Chars(suppl):
    char_names = ['Chi0','Chi0n','Chi0v','Chi1',\
    'Chi1n','Chi1v','Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v',\
    'EState_VSA1','EState_VSA10','EState_VSA11','EState_VSA2','EState_VSA3',\
    'EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8',\
    'EState_VSA9','FractionCSP3','HallKierAlpha','HeavyAtomCount','Ipc',\
    'Kappa1','Kappa2','Kappa3','LabuteASA','MolLogP','MolMR','MolWt',\
    'NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles',\
    'NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles',\
    'NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms',\
    'NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles',\
    'NumSaturatedRings','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12',\
    'PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5',\
    'PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','RingCount','SMR_VSA1',\
    'SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6','SMR_VSA7',\
    'SMR_VSA8','SMR_VSA9','SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12',\
    'SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7',\
    'SlogP_VSA8','SlogP_VSA9','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',\
    'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',\
    'VSA_EState8','VSA_EState9']
    calc = MolecularDescriptorCalculator(char_names)
    full_list = []

    for mol in suppl:
        if mol is None: continue
        #row_list = [mol.GetProp('_Name')]
        full_list.append([mol.GetProp('_Name')]+list(calc.CalcDescriptors(mol)))
    return([['_Name']+char_names]+(full_list))

def Read_Process_Save(expr_loc,path_to_sdf,save_path):

    expr_table = pd.read_csv(expr_loc)
    
    # this deals out SD formatted molecules
    suppl = Chem.SDMolSupplier(path_to_sdf)
    #remove repeats
    expr_table,dropped_cids = Consistant_Checker(expr_table.loc[:,'PUBCHEM_CID':'PUBCHEM_ACTIVITY_OUTCOME'])
    #get mfps
    fp_length = 1024
    fp_list = get_mfp_hash(fp_length,suppl)
    #get mol_chars
    mol_chars = Add_Mol_Chars(suppl)
    #combine the labels and the finger prints
    activity_table = Align_with_Activity(expr_table,fp_list,dropped_cids)
    #combine the labels and the mol_chars
    activity_table = Align_with_Activity(activity_table,mol_chars,dropped_cids)
    pickle_on = open(save_path,'wb')
    pickle.dump(activity_table,pickle_on)
    pickle_on.close()
path_lists = [[r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_628_datatable.csv',
 r'C:\Users\gdrei\Downloads\3283798507800496206.sdf\3283798507800496206.sdf',
 r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_628_processed.pkl'],[r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_894_datatable.csv',
 r'C:\Users\gdrei\Downloads\2616060013533693584.sdf\2616060013533693584.sdf',
 r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_894_processed.pkl'],[r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_449739_datatable.csv',
 r'C:\Users\gdrei\Downloads\1441395869637736289.sdf\1441395869637736289.sdf',
 r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_449739_processed.pkl']]
for experiment in path_lists:
    [expr_loc,path_to_sdf,save_path] = experiment[:]
    expr_table = pd.read_csv(expr_loc)
    
    # this deals out SD formatted molecules
    suppl = Chem.SDMolSupplier(path_to_sdf)
    #remove repeats
    expr_table,dropped_cids = Consistant_Checker(expr_table.loc[:,'PUBCHEM_CID':'PUBCHEM_ACTIVITY_OUTCOME'])
    #get mfps
    fp_length = 1024
    fp_list = get_mfp_hash(fp_length,suppl)
    #get mol_chars
    mol_chars = Add_Mol_Chars(suppl)
    #combine the labels and the finger prints
    activity_table = Align_with_Activity(expr_table,fp_list,dropped_cids)
    #combine the labels and the mol_chars
    activity_table = Align_with_Activity(activity_table,mol_chars,dropped_cids)
    pickle_on = open(save_path,'wb')
    pickle.dump(activity_table,pickle_on)
    pickle_on.close()