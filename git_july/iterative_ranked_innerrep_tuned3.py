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
exp.log_other('Hypothesis','''15% start, 5% iter, all random, svm hinge loss and sigmoidinstead of calibrated cv''')
exper_file_name = 'tuned_3_svmmod_sigmoid'
import pickle, sys, os
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import numpy as np
from Iterative_help_funcs_tuned import get_Scaled_Data,train_SVM,train_DNN,train_RF,train_LGBM,calc_and_save_metrics,train_PyTorchGCNN,train_random_classifier
from imblearn.over_sampling import RandomOverSampler
#choosing a 3:1 Inactive to Active ratio
ros = RandomOverSampler(sampling_strategy= 0.33)
import pandas as pd
from joblib import Parallel, delayed
#from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
    
#%% 
'''Load data'''

AID_list =['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_596','AID_893','AID_894']
#AID_list =['AID_1345083','AID_624255','AID_449739']

#This will hold a series of dicts of metrics that we then build into a dataframe

#using longest most imformantive embeddings
      
classifier_dict = {'SVM': train_SVM, 'RF': train_RF, 'LGBM':train_LGBM,'DNN':train_DNN,'GCNN_pytorch':train_PyTorchGCNN,'random':train_random_classifier}

model_list = ['SVM','RF','random']
#model_list = ['RF']
#model_list = ['GCNN_pytorch']
#model_list = ['SVM']

num_models = len(model_list)
mmp = MaxMinPicker()               
def getRandomIterInds(firstPicksList,fplist,bottom_to_select):
    full_list_index = np.arange(len(fplist))
    unselected_inds = list(set(full_list_index) - set(firstPicksList))
    random_selection = np.random.choice(unselected_inds,bottom_to_select,replace=False)
    start_indexs = np.concatenate((firstPicksList,random_selection),axis=0)
    return start_indexs
def getNextIterInds(firstPicksList,fplist,bottom_to_select):
    diverse_picks = mmp.LazyBitVectorPick(fplist,len(fplist),len(firstPicksList)+bottom_to_select,firstPicksList)
    start_indexs = np.array(diverse_picks)
    return start_indexs
#define size of iter after first 10% train relative to that trainsize
iterRel2Start = 0.5
end_iter = 9
for repetition_number in range(3):
    multi_dump_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', exper_file_name+str(repetition_number)+'.pkl') 
    exp.log_other('Metrics Dict Path_'+str(repetition_number),multi_dump_path)

metric_dict_metalist =[[],[],[]]
    
for AID in AID_list:
        print(AID)
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
        #build up metalists that will vary for each repetition
        start_num = int(len(fplist)*0.15)
        with parallel_backend('multiprocessing'):
            start_index_metalist = Parallel(n_jobs=3)(delayed(getNextIterInds)(firstPicksList=i, fplist=j,bottom_to_select=k) for i,j,k in zip([[],[],[]],[fplist,fplist,fplist],[start_num,start_num,start_num]))
#            start_indexs = np.array(mmp.LazyBitVectorPick(fplist,len(fplist),int(len(fplist)/10)))
        '''store in a list that will vary as each model makes its predictions'''
        for rep_num in range(3):
            #set rep specific variables
            metric_dict_list = metric_dict_metalist[rep_num]
            start_indexs = start_index_metalist[rep_num]
            multi_dump_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', exper_file_name+str(rep_num)+'.pkl') 
            #everything from here forwards doesn't need to change
            start_ind_list=[start_indexs for i in range(num_models)]
            diverse_size_list = [0 for i in range(num_models)]
            fp_metalist = [fplist for i in range(num_models)]
            library_size = len(fplist)
            iter_num = 0 
            while iter_num < end_iter:
                print("Beginning Iteration ",iter_num)
                if iter_num < 5:
                    selection_type = 'Random'
                else:
                    selection_type = 'Random'
                        
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
                        #have this split here so that I can deal w fact that DNNs
                        #are now returning the history
                        if model_type =='DNN' or model_type=='GCNN_pytorch':
                            train_predicted_probs,test_predicted_probs,base_test_predicted_probs,hist = train_and_predict_model(over_X_train,X_test,over_y_train,y_test,X_train)
                        else:
                            train_predicted_probs,test_predicted_probs,base_test_predicted_probs = train_and_predict_model(over_X_train,X_test,over_y_train,y_test,X_train)                    
                            hist = None
                        metric_dict_list = calc_and_save_metrics(y_test,test_predicted_probs,model_type,
                                              embedding_type,AID,metric_dict_list,iter_num,'test',hist)
                        metric_dict_list = calc_and_save_metrics(over_y_train,train_predicted_probs,model_type,
                                              embedding_type,AID,metric_dict_list,iter_num,'train',hist)
                        metric_dict_list = calc_and_save_metrics(y_train,base_test_predicted_probs,model_type,
                                              embedding_type,AID,metric_dict_list,iter_num,'base_train',hist)
                        '''Now select next 5% section'''
                        '''Put labels and preds in df, sort them. take top 80% of tier size of the top predictions
                        then do a diverse selection or random selection for remaining 20% more'''
                        preds_df = pd.DataFrame({'activity_table_index':np.array(test_index),'prob_active':np.array(test_predicted_probs)},columns= np.array(['activity_table_index','prob_active']))
                        preds_df.sort_values('prob_active',ascending=False,inplace=True,axis=0)
                        next_inds=[]
                        top_to_select = int(len(activity_table)*0.04)
                        explore_select = int(len(activity_table)*0.01)
                        next_inds=next_inds+preds_df.head(top_to_select)['activity_table_index'].tolist()
                        firstPicksList = next_inds+(start_indexs.tolist())
                        start_ind_list[list_idx] = firstPicksList
                        diverse_size_list[list_idx] = explore_select
    

                with parallel_backend('multiprocessing'):
                    if selection_type == 'Diverse':
                        start_ind_list = Parallel(n_jobs=num_models)(delayed(getNextIterInds)(firstPicksList=i, fplist=j,bottom_to_select=k) for i,j,k in zip(start_ind_list,fp_metalist,diverse_size_list))
                    elif selection_type == 'Random':
                        start_ind_list = Parallel(n_jobs=num_models)(delayed(getRandomIterInds)(firstPicksList=i, fplist=j,bottom_to_select=k) for i,j,k in zip(start_ind_list,fp_metalist,diverse_size_list))
    
                metrics_df = pd.DataFrame(metric_dict_list)
                metrics_df.to_pickle(multi_dump_path)
                iter_num +=1
exp.end()
