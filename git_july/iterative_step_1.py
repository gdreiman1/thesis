#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:31:16 2019

@author: gabriel
"""

'''Start of iterative process thing'''

'''Select data set, do smart sampling either rdkit: https://www.rdkit.org/docs/source/rdkit.ML.Cluster.Butina.html
https://squonk.it/docs/cells/RDKit%20MaxMin%20Picker/
or from deep chem: MaxMinSplitter(Splitter), ButinaSplitter(Splitter), FingerprintSplitter(Splitter)

can use maxmin splitter to fill out remainining space in next iter by specifying number remaining as sample size and the pool of 
compounds as target
'''
#%% 
'''import'''
import pickle, sys, os
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import numpy as np
from Iterative_help_funcs import get_Scaled_Data,train_SVM,train_DNN,train_RF,train_LGBM,calc_and_save_metrics
from imblearn.over_sampling import RandomOverSampler
#choosing a 3:1 Inactive to Active ratio
ros = RandomOverSampler(sampling_strategy= 0.33)
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
#%% 
'''Load data'''

AID_list =['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']
#This will hold a series of dicts of metrics that we then build into a dataframe
metric_dict_list = []
#using longest most imformantive embeddings
embedding_type = 'MFPMolChars'        
classifier_dict = {'SVM': train_SVM, 'RF': train_RF, 'LGBM':train_LGBM,'DNN':train_DNN}
model_list = ['SVM','RF','LGBM','DNN']
mmp = MaxMinPicker()
for AID in AID_list:
#    for model_type in ['SVM']:
    if 'win' in sys.platform:
        AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
    else:
        AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
    save_path = AID_path+ '/' + AID +'_processed.pkl'
    pickle_off = open(save_path,'rb')
    activity_table=pickle.load(pickle_off)
    pickle_off.close()
    
    '''Pick diverse starting point of 10% of library'''
    fplist = [x for x in activity_table['bit_MFP']]
    start_indexs = np.array(mmp.LazyBitVectorPick(fplist,len(fplist),int(len(fplist)/10)))
    '''store in a list that will vary as each model makes its predictions'''
    start_ind_list=[start_indexs for i in range(4)]
    diverse_size_list = [0,0,0,0]
    fp_metalist = [fplist for i in range(4)]
    iter_num = 0 
    while iter_num < 5:
        print("Beginning Iteration ",iter_num)
        '''run thru 4 models and get their preds for this iter'''
        for list_idx,[model_type,start_indexs] in enumerate(zip(model_list,start_ind_list)):
            '''Get data for the starting molecules'''
            test_index = list(set(activity_table.index)-set(start_indexs))
            X_train,X_test,y_train,y_test = get_Scaled_Data(start_indexs,test_index,activity_table,True,'MFPMolChars')
            #oversample to 2:1 Inactive to Active
            '''what happens when ratio is better than ros???'''
            over_X_train,over_y_train = ros.fit_resample(X_train,y_train)
            '''Inital train run'''
            train_and_predict_model = classifier_dict[model_type]
            train_predicted_probs,test_predicted_probs = train_and_predict_model(over_X_train,X_test,over_y_train,y_test)
            
            metric_dict_list = calc_and_save_metrics(y_test,test_predicted_probs,model_type,
                                  embedding_type,AID,metric_dict_list,iter_num,'test')
            metric_dict_list = calc_and_save_metrics(over_y_train,train_predicted_probs,model_type,
                                  embedding_type,AID,metric_dict_list,iter_num,'train')
            '''Now select next 10% section'''
            '''Put labels and preds in df, sort them. take top 700 greater than zero, bottom 100 greater than zero, top 100 less than zero, 
            then do a diverse selection for 100 more'''
            preds_df = pd.DataFrame({'activity_table_index':np.array(test_index),'prob_active':np.array(test_predicted_probs)},columns= np.array(['activity_table_index','prob_active']))
            preds_df.sort_values('prob_active',ascending=False,inplace=True,axis=0)
            #split at 0.5 Could later runanalysis to calibrate descision boundary
            pos_preds_df = preds_df.loc[preds_df['prob_active']>=0.5]
            neg_preds_df = preds_df.loc[preds_df['prob_active']<0.5]
            num_pred_active = len(pos_preds_df)
            next_inds=[]
            #if we have enought to get top 7% and bottom 1%
            if num_pred_active >= int(len(activity_table)*0.08):
                top_to_select = int(len(activity_table)*0.07)
                bottom_to_select = int(len(activity_table)*0.01)
                next_inds=next_inds+pos_preds_df.head(top_to_select)['activity_table_index'].tolist()
                next_inds=next_inds+pos_preds_df.tail(bottom_to_select)['activity_table_index'].tolist()
                next_inds=next_inds+neg_preds_df.head(bottom_to_select)['activity_table_index'].tolist()
            else:
                #even split between weak inactives and random divers sample
                next_inds=next_inds+pos_preds_df['activity_table_index'].tolist()
                bottom_to_select = int((len(activity_table)*0.1-len(next_inds))*0.5)
                next_inds=next_inds+neg_preds_df.head(bottom_to_select)['activity_table_index'].tolist()
            #make list of all indexs currently selected
            firstPicksList = next_inds+(start_indexs.tolist())
            start_ind_list[list_idx] = firstPicksList
            diverse_size_list[list_idx] = bottom_to_select
        '''This is a slow single core process so we're gonna parralelize it'''
    
        def getNextIterInds(fistPicksList,fplist,bottom_to_select):
            diverse_picks = mmp.LazyBitVectorPick(fplist,len(fplist),len(firstPicksList)+bottom_to_select,firstPicksList)
            start_indexs = np.array(diverse_picks)
            return start_indexs
        with parallel_backend('multiprocessing'):
            start_ind_list = Parallel(n_jobs=4)(delayed(getNextIterInds)(fistPicksList=i, fplist=j,bottom_to_select=k) for i,j,k in zip(start_ind_list,fp_metalist,diverse_size_list))
        iter_num +=1
metrics_df = pd.DataFrame(metric_dict_list)
multi_dump_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', 'first_iter_run.pkl') 
metrics_df.to_pickle(multi_dump_path)
