#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:07:05 2019

@author: gabriel
"""

'''Creating a script to tune hyper parameters for each classifier!'''
#from comet_ml import Experiment
#exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
#                        project_name="hyperparameter_tuning", workspace="gdreiman1", disabled = False)
#exp.log_code = True
import pickle, sys, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, auc
import lightgbm as lgb
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from imblearn.over_sampling import RandomOverSampler
from joblib import Parallel, delayed
from joblib import parallel_backend


#%%
# take in all AID's and store in  a dict
AID_list =['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_596','AID_893','AID_894']
AID_dict = {}
for AID in AID_list:
        print(AID)
        if 'win' in sys.platform:
            AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
        else:
            AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
        save_path = AID_path+ '/' + AID +'_processed.pkl'
        pickle_off = open(save_path,'rb')
        activity_table=pickle.load(pickle_off)
        pickle_off.close()
        
        AID_dict[AID] = activity_table
        
#%%
'''helper funcs'''
mmp = MaxMinPicker()
ros = RandomOverSampler(sampling_strategy= 0.33)
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs = -1)

svm =  SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=500000, 
                                                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
                                                        validation_fraction=0.1, n_iter_no_change=5, class_weight='balanced', warm_start=False, average=False)
lgbm =  lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=500, subsample_for_bin=200000, 
                                        objective='binary', is_unbalance=True, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
                                        subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, 
                                        importance_type='split')
classifier_dict ={'SVM':svm,'LGBM':lgbm,'RF':rf}
embedding_type = 'MFPMolChars'
rf_space  =[Categorical([1, 10,20, None], name='max_depth'),
          Categorical(['balanced','balanced_subsample'],name='class_weight'),
          Categorical([200,400,800,1600,3200], name='n_estimators'),
          Categorical(['auto','log2'], name='max_features'),
          Categorical([True,False], name= 'bootstrap'),
          Integer(2,10, name='min_samples_split'),
          Integer(1,5, name='min_samples_leaf')]
 
lgbm_space  = [Categorical([1,5,10,15,20, -1], name='max_depth'),
               Integer(5,50, name='num_leaves'),
               Categorical(['gbdt','dart'],name='boosting_type'),
               Categorical([True,False],name='is_unbalance'),

              Categorical([200,400,800,1600,3200], name='n_estimators'),
              Real(0.01,0.5, name='learning_rate'),
              Integer(25,500, name= 'max_bin'),
              Integer(10,50, name='min_data_in_leaf')]  
svm_space = [Categorical(['optimal','invscaling'], name='learning_rate'),
               Integer(500,100000, name='max_iter'),
               Categorical(['hinge',],name='loss'),
               Categorical(['l2','elasticnet'],name='penalty'),
              Real(0.00001,0.001, name='eta0'),

              Real(0.00001,0.001, name='alpha'),
              Categorical(['balanced',None], name= 'class_weight'),
              Categorical([False,5,10,20,True], name='average')]
              
space_dict = {'SVM':svm_space,'LGBM':lgbm_space,'RF':rf_space}             
def getNextIterInds(firstPicksList,fplist,bottom_to_select):
    diverse_picks = mmp.LazyBitVectorPick(fplist,len(fplist),len(firstPicksList)+bottom_to_select,firstPicksList)
    start_indexs = np.array(diverse_picks)
    return start_indexs
def getDiverseSplits(num_cv,fplist,start_num):
    firstPicksList = [[] for x in range(num_cv)]
    fp_metalist = [fplist for x in range(num_cv)]
    start_numlist = [start_num for x in range(num_cv)]
    with parallel_backend('multiprocessing'):
        start_ind_list = Parallel(n_jobs=-1)(delayed(getNextIterInds)(firstPicksList=i, fplist=j,bottom_to_select=k) for i,j,k in zip(firstPicksList,fp_metalist,start_numlist))
    return start_ind_list
class Scorer:
    '''A class that simplifies calculating score for a single AID. Takes in AID
    dataframe and calculates a fixed number of 10% starting indexs. Then for a 
    particular classifier setting, takes the classifier and calculates metrics on it
    for num_cv iterations'''
    def __init__(self,activity_table,num_cv,num_starts):
        self.activity_table = activity_table
        self.num_cv = num_cv
        self.num_starts = num_starts
    def storeDiverseStarts(self):
        fplist = [x for x in self.activity_table['bit_MFP']]
        start_num = int(len(fplist)/10)
        start_index_list = getDiverseSplits(self.num_starts,fplist,start_num)
        self.start_index_list = start_index_list
        
    def getScore(self,classifier, classifier_name):
        score = 0
        #get cv_num random diverse points
        #basically randomly generating indexs to the list of start inds that we
        #store in storeDiverseStarts
        cv_index_list = [self.start_index_list[x] for x in np.random.randint(self.num_starts,size=self.num_cv)]
        #iterate the splits scaling and oversampling, collect scores
        for train_inds in cv_index_list:
            test_index = list(set(self.activity_table.index)-set(train_inds))
            X_train,X_test,y_train,y_test = get_Scaled_Data(train_inds,test_index,self.activity_table,True,embedding_type)
            over_X_train,over_y_train = ros.fit_resample(X_train,y_train)
            classifier.fit(over_X_train,over_y_train)
            #used calibrated SVM to get preds
            if classifier_name == 'SVM':
                cv_model = CalibratedClassifierCV(classifier,'sigmoid','prefit')
                cv_model.fit(over_X_train,over_y_train)
                classifier=cv_model
            pred_probs = classifier.predict_proba(X_test)
            prec_array,recall_array,thresh_array = precision_recall_curve(y_test,pred_probs[:,1])
            auc_PR = auc(recall_array,prec_array)
            score += auc_PR
        return -score/self.num_cv

def get_Scaled_Data(train_ind,test_ind,activity_table,bin_labels,embedding_type):
    
    #get start and end index for molchars
    MC_start = activity_table.columns.get_loc('Chi0')
    #need to add 1 bc exclusive indexing
    MC_end = activity_table.columns.get_loc('VSA_EState9')+1
    # standardize data    
    scaler = StandardScaler(copy = False)
    #get MFP in right order
    fp_length = len(activity_table.iloc[5]['MFP'])
    #reshape mfp
    X_mfp = np.concatenate(np.array(activity_table['MFP'])).ravel()
    X_mfp = X_mfp.reshape((-1,fp_length))
    #return requested datatype
    if embedding_type == 'MFPMolChars':
        X_train_molchars_std = scaler.fit_transform(np.array(activity_table.iloc[train_ind,MC_start:MC_end]).astype(float))
        X_test_molchars_std = scaler.transform(np.array(activity_table.iloc[test_ind,MC_start:MC_end]).astype(float))
        X_train = np.concatenate((X_mfp[train_ind,:],X_train_molchars_std),axis = 1)
        X_test = np.concatenate((X_mfp[test_ind,:],X_test_molchars_std),axis = 1)
    elif embedding_type == 'MFP':
        X_train = X_mfp[train_ind,:]
        X_test = X_mfp[test_ind,:]
    elif embedding_type == 'MolChars':
        X_train_molchars_std = scaler.fit_transform(np.array(activity_table.iloc[train_ind,MC_start:MC_end]).astype(float))
        X_test_molchars_std = scaler.transform(np.array(activity_table.iloc[test_ind,MC_start:MC_end]).astype(float))
        X_train = X_train_molchars_std
        X_test = X_test_molchars_std
#    elif embedding_type == 'Graph':
#        '''Getting graph reps from mol2graph that I stored in df'''
#        X_train = activity_table['Graph Rep'].iloc[train_ind].tolist()
#        X_test = activity_table['Graph Rep'].iloc[test_ind].tolist()
#        y_train = activity_table['PUBCHEM_ACTIVITY_OUTCOME'].iloc[train_ind]
#        y_test = activity_table['PUBCHEM_ACTIVITY_OUTCOME'].iloc[test_ind]
##       remapping active to 1 and everything else to zero
#        bin_y_train, bin_y_test = np.array([1 if x == 'Active' else 0 for x in y_train]),np.array([1 if x =='Active'  else 0 for x in y_test])
#        for data, label in zip(X_train,bin_y_train):
#            data.y = torch.tensor([[label]],dtype=torch.float)
#        for data, label in zip(X_test,bin_y_test):
#            data.y = torch.tensor([[label]],dtype=torch.float)

    y_train = activity_table['PUBCHEM_ACTIVITY_OUTCOME'].iloc[train_ind]
    y_test = activity_table['PUBCHEM_ACTIVITY_OUTCOME'].iloc[test_ind]
    #remapping active to 1 and everything else to zero
    bin_y_train, bin_y_test = np.array([1 if x == 'Active' else 0 for x in y_train]),np.array([1 if x =='Active'  else 0 for x in y_test])
    if bin_labels==True:
        y_test = bin_y_test
        y_train = bin_y_train
    return X_train,X_test,y_train,y_test
#%%
import time

'''Do the actual testing'''
opt_results_list = []
for AID, activity_table in AID_dict.items():        
    current_scorer = Scorer(activity_table,4,20)
    current_scorer.storeDiverseStarts()
    for classifier_name, classifier in classifier_dict.items():
#        if classifier_name == 'RF':
        space = space_dict[classifier_name]
        print(AID)
        t0 =time.time()
        @use_named_args(space)
        def objective(**params):
            classifier.set_params(**params)
            score = current_scorer.getScore(classifier,classifier_name) 
            return score
        opt_res = gp_minimize(objective, space, n_calls=50,n_random_starts = 10, verbose=0)
        opt_results_list.append({'AID':AID,'Classifier':classifier_name,'opt_res':opt_res})
        t1=time.time()
        print(AID,' ',classifier_name,' time: ',t1-t0)
    
        opt_df = pd.DataFrame(opt_results_list)
        dump_path ='/home/gabriel/Dropbox/UCL/Thesis/Data/hypertune_50.csv'
        opt_df.to_csv(dump_path)
#exp.end()