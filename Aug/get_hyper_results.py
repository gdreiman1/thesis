#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:10:59 2019

@author: gabriel
"""

'''Reading out hyperparameter tuning results'''
import pandas as pd
import ast
svm_lgbm_df =  pd.read_csv('/home/gabriel/Dropbox/UCL/Thesis/Data/hypertune_SVM_LGBM3.csv')
svm_list = []
for _,row in svm_lgbm_df[svm_lgbm_df['Classifier']=='SVM'].iterrows():
    opt_res = row['opt_res']
    x_loc = opt_res.find('x: [')
    info_start = opt_res.find('[',x_loc)
    info_end = opt_res.find(']',x_loc)
    data = ast.literal_eval(opt_res[info_start:info_end+1])
    data = [row['AID']]+data
    svm_list.append(data)
svm_data = pd.DataFrame.from_records(svm_list,columns=['AID','learning_rate','max_iter','loss','penalty','eta0','alpha','class_weight','average'])
lgbm_list = []
for _,row in svm_lgbm_df[svm_lgbm_df['Classifier']=='LGBM'].iterrows():
    opt_res = row['opt_res']
    x_loc = opt_res.find('x: [')
    info_start = opt_res.find('[',x_loc)
    info_end = opt_res.find(']',x_loc)
    data = ast.literal_eval(opt_res[info_start:info_end+1])
    data = [row['AID']]+data
    lgbm_list.append(data)
lgbm_data = pd.DataFrame.from_records(lgbm_list,columns=['AID','max_depth','num_leaves','boosting_type',
                                                         'is_unbalance','n_estimators','learning_rate','max_bin','min_data_in_leaf',
                                                         ])
rf_df =  pd.read_csv('/home/gabriel/Dropbox/UCL/Thesis/Data/hypertune_rf.csv')
rf_list = []
for _,row in rf_df[rf_df['Classifier']=='RF'].iterrows():
    opt_res = row['opt_res']
    x_loc = opt_res.find('x: [')
    info_start = opt_res.find('[',x_loc)
    info_end = opt_res.find(']',x_loc)
    data = ast.literal_eval(opt_res[info_start:info_end+1])
    data = [row['AID']]+data

    rf_list.append(data)
rf_data = pd.DataFrame.from_records(rf_list,columns=['AID','max_depth','class_weight','n_estimators',
                                                     'max_features','bootstrap','min_samples_split',
                                                     'min_samples_leaf'])
    
    print(rf_data.to_latex())
    print(svm_data.to_latex())
    print(rf_data.to_latex())

#find mode
#
#make those settings in the new help functions file
#
#make appropriate mods in the tuned1 etc files
#run them
