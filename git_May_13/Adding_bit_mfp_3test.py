# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:54:59 2019

@author: gdrei
"""
'''ADDING PUB_CHEM_ACTIVITY_SCORE to 6 inital files. Also remaining 2D descriptors'''


import glob
import os,sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw,rdMolDescriptors
import pickle
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from joblib import Parallel, delayed
def AddBitVec(AID):
    if 'win' in sys.platform:
        AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
    else:
        AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID)
    path_to_sdf = glob.glob(AID_path+'/*.sdf')[0]
    expr_loc = glob.glob(AID_path+'/*.csv')[0]
    save_path = AID_path+ '/' + AID +'_processed.pkl'
    
    # this deals out SD formatted molecules
    suppl = Chem.SDMolSupplier(path_to_sdf)
#    expr_table = pd.read_csv(expr_loc)
#    #drop all nans
#    nan_inds = expr_table.index[expr_table['PUBCHEM_CID'].isna()].tolist()
#    expr_table.drop(nan_inds,inplace = True)
    #
    pickle_off = open(save_path,'rb')
    activity_table=pickle.load(pickle_off)
    pickle_off.close()
    #convert the cids to str
#    expr_table['PUBCHEM_CID'] =expr_table['PUBCHEM_CID'].astype(int).astype(str)
#    # add the activity score
#    a = expr_table.loc[:,['PUBCHEM_CID','PUBCHEM_ACTIVITY_SCORE']]
#    activity_table = activity_table.merge(a, on='PUBCHEM_CID')
#    #pop out column
#    ordered_activity = activity_table.pop('PUBCHEM_ACTIVITY_SCORE')
#    #insert it in the 2nd index
#    activity_table.insert(2,'PUBCHEM_ACTIVITY_SCORE',ordered_activity)
    '''Add explicit bitvector '''#    additional_features = Add_Mol_Chars_Only_3_Feats(suppl)
#    activity_table = activity_table.merge(additional_features, on='PUBCHEM_CID')
    #build dict list
    bit_MFP_dict = [{'PUBCHEM_CID':mol.GetProp('_Name'),'bit_MFP':rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,2,nBits = 1024)} for mol in suppl]
    bit_df = pd.DataFrame(bit_MFP_dict)
    #merge it
    activity_table = activity_table.merge(bit_df, on='PUBCHEM_CID')
    pickle_on = open(save_path,'wb')
    pickle.dump(activity_table,pickle_on)
    pickle_on.close()
        
#for AID in ['AID_628','AID_894','AID_449739','AID_624255','AID_1345083']:
AID_list = ['AID_1259354', 'AID_598', 'AID_488969']    
Parallel(n_jobs=8)(delayed(AddBitVec)(AID) for AID in AID_list)