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
from comet_ml import Experiment
exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                        project_name="iter_baseline", workspace="gdreiman1", disabled = False
                        )
exp.log_code = True
exp.log_other('Hypothesis','''Only scan predicted actives,just runnign tf so see what happens''')
import pickle, sys, os
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import numpy as np
from Iterative_help_funcs import get_Scaled_Data,train_SVM,train_DNN,train_RF,train_LGBM,calc_and_save_metrics,train_PyTorchGCNN
from imblearn.over_sampling import RandomOverSampler
#choosing a 3:1 Inactive to Active ratio
ros = RandomOverSampler(sampling_strategy= 0.33)
import pandas as pd
#from joblib import Parallel, delayed
#from joblib.externals.loky import set_loky_pickler
#from joblib import parallel_backend
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

#%% 
'''Load data'''

AID_list =['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_596','AID_893','AID_894']
#AID_list =['AID_893','AID_894']

#This will hold a series of dicts of metrics that we then build into a dataframe
metric_dict_list = []
multi_dump_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', 'second_greedy_TF_iter_run.pkl') 
exp.log_other('Metrics Dict Path',multi_dump_path)
#using longest most imformantive embeddings
      
classifier_dict = {'SVM': train_SVM, 'RF': train_RF, 'LGBM':train_LGBM,'DNN':train_DNN,'GCNN_pytorch':train_PyTorchGCNN}

model_list = ['DNN']
num_models = len(model_list)
mmp = MaxMinPicker()
#define how we select after inital training run
selection_type = 'Diverse'
#define size of iter after first 10% train relative to that trainsize
iterRel2Start = 0.5
end_iter = 10
for AID in AID_list:
        if 'win' in sys.platform:
            AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
        else:
            AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
        save_path = AID_path+ '/' + AID +'graph_processed.pkl'
        pickle_off = open(save_path,'rb')
        activity_table=pickle.load(pickle_off)
        pickle_off.close()
        
        '''Pick diverse starting point of 10% of library'''
        fplist = [x for x in activity_table['bit_MFP']]
        '''start_indexs holds the indexes of molecules already scanned at the
        start of each iteration. So for the first iter it hold the diversity selection. 
        For the second, it holds both the diversity selection and the molecules 
        screened based on the results of the first training iteration etc'''
        start_indexs = np.array(mmp.LazyBitVectorPick(fplist,len(fplist),int(len(fplist)/10)))
        '''store in a list that will vary as each model makes its predictions'''
        start_ind_list=[start_indexs for i in range(num_models)]
        diverse_size_list = [0 for i in range(num_models)]
        fp_metalist = [fplist for i in range(num_models)]
        library_size = len(fplist)
        iter_num = 0 
        while iter_num < end_iter:
            print("Beginning Iteration ",iter_num)
            '''run thru models and get their preds for this iter'''
            for list_idx,[model_type,start_indexs] in enumerate(zip(model_list,start_ind_list)):
                '''Get data for the starting molecules, it will be graphs for GCNN, else the MFP_MolChars'''
                test_index = list(set(activity_table.index)-set(start_indexs))
                #check that we haven't exceeded 50% of library
                if len(test_index) > int(0.5 *library_size):
                    if model_type == 'GCNN_pytorch':
                        embedding_type = 'Graph'
                        X_train,X_test,y_train,y_test = get_Scaled_Data(start_indexs,test_index,activity_table,True,embedding_type)
                    else:
                        embedding_type = 'MFPMolChars'                  
                        X_train,X_test,y_train,y_test = get_Scaled_Data(start_indexs,test_index,activity_table,True,embedding_type)
                    #oversample to 2:1 Inactive to Active
                    '''what happens when ratio is better than ros??? Bad boy errors!!
                    Now need another check to get out of our pkl with over enrichment (hahaha)'''
                    if (len(y_train)/sum(y_train))<0.25:
                    #need to ros iff ratio is less than 3:1
                        if model_type == 'GCNN_pytorch':
                            '''ros doesn't like the data apparently'''
                            over_X_train,over_y_train = ros.fit_resample(np.arange(len(X_train)).reshape((-1,1)),y_train)
                            over_X_train = over_X_train.reshape(-1)
                            over_X_train = [X_train[i] for i in over_X_train]
                        else:
                            over_X_train,over_y_train = ros.fit_resample(X_train,y_train)
                    else:
                    #just use current enriched sample
                            over_X_train,over_y_train = X_train,y_train
                    '''Inital train run'''
                    train_and_predict_model = classifier_dict[model_type]
                    #now I'm saving the history from the NNs
                    if model_type =='DNN' or model_type=='GCNN_pytorch':
                        train_predicted_probs,test_predicted_probs,base_test_predicted_probs,hist = train_and_predict_model(over_X_train,X_test,over_y_train,y_test,X_train)
                    else:
                        train_predicted_probs,test_predicted_probs,base_test_predicted_probs = train_and_predict_model(over_X_train,X_test,over_y_train,y_test,X_train)

                    metric_dict_list = calc_and_save_metrics(y_test,test_predicted_probs,model_type,
                                          embedding_type,AID,metric_dict_list,iter_num,'test')
                    metric_dict_list = calc_and_save_metrics(over_y_train,train_predicted_probs,model_type,
                                          embedding_type,AID,metric_dict_list,iter_num,'train')
                    metric_dict_list = calc_and_save_metrics(y_train,base_test_predicted_probs,model_type,
                                          embedding_type,AID,metric_dict_list,iter_num,'base_train')
                    '''Now select next 10% section'''
                    '''Put labels and preds in df, sort them. take top 700 greater than zero, bottom 100 greater than zero, top 100 less than zero, 
                    then do a diverse selection for 100 more'''
                    preds_df = pd.DataFrame({'activity_table_index':np.array(test_index),'prob_active':np.array(test_predicted_probs)},columns= np.array(['activity_table_index','prob_active']))
                    preds_df.sort_values('prob_active',ascending=False,inplace=True,axis=0)
                    #split at 0.5 Could later runanalysis to calibrate descision boundary
                    pos_preds_df = preds_df.loc[preds_df['prob_active']>=0.5]
                    neg_preds_df = preds_df.loc[preds_df['prob_active']<0.5]
                    num_pred_active = len(pos_preds_df)
                    
                    if num_pred_active <= int(0.5 *library_size):
                        next_inds = pos_preds_df['activity_table_index'].tolist()
                    else:
                        next_inds = pos_preds_df['activity_table_index'].tolist()
                        #select remaining space for data
                        pred_space_left = int(library_size*0.5-len(start_indexs))
                        next_inds = next_inds[:pred_space_left]
                    firstPicksList = next_inds+(start_indexs.tolist())
                    start_ind_list[list_idx] = np.array(firstPicksList)
                    metrics_df = pd.DataFrame(metric_dict_list)
                    metrics_df.to_pickle(multi_dump_path)
            iter_num +=1
exp.end()
