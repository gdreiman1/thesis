# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:57:18 2019

@author: gdrei
"""

from Data_ProcessandAlign import Read_Process_Save
import glob
import os

for AID in ['AID_1259354', 'AID_598', 'AID_488969']:
    
    AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data', AID) 
    sdf_path = glob.glob(AID_path+'/*.sdf')
    expr_loc = glob.glob(AID_path+'/*.csv')
    save_path = AID_path+ '/' + AID +'_processed.pkl'
    Read_Process_Save(expr_loc[0],sdf_path[0],save_path)
