# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:17:42 2019

@author: gdrei

Just storing random snippt
"""

def Get_Suppl_Mol_Des(suppl):
    from joblib import Parallel, delayed 
    Mol_Desc_list = Parallel(n_jobs=4)(delayed(Get_Single_Mol_Descripts)(mol) for mol in suppl)
    return(Mol_Desc_list)
def Get_Single_Mol_Descripts(mol):
    name_list = ['_Name','Chi0','Chi0n','Chi0v','Chi1',\
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
    return([mol.GetProp(name_list[i]) for i in range(len(name_list))])