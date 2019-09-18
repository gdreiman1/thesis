# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:37 2019

@author: gdrei
"""
from comet_ml import Experiment
experiment = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                        project_name="general_test", workspace="gdreiman1")
experiment.log_code = True
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
#%%
def comet_SVM(save_path):
#encode labels
pickle_off = open(save_path,'rb')
activity_table=pickle.load(pickle_off)
pickle_off.close()
#get length of MFP
fp_length = len(activity_table.iloc[5]['MFP'])

le = sklearn.preprocessing.LabelEncoder()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy = False)
labels = le.fit_transform(activity_table['PUBCHEM_ACTIVITY_OUTCOME'])
#split data:
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, train_size=None, random_state=2562)
X_mfp = np.concatenate(np.array(activity_table['MFP'])).ravel()
X_mfp = X_mfp.reshape((-1,fp_length))
for train_ind, test_ind in splitter.split(X_mfp,labels):
    # standardize data
    X_train_molchars_std = scaler.fit_transform(np.array(activity_table.iloc[train_ind,4:]))
    X_test_molchars_std = scaler.transform(np.array(activity_table.iloc[test_ind,4:]))
    X_train = np.concatenate((X_mfp[train_ind,:],X_train_molchars_std),axis = 1)
    X_test = np.concatenate((X_mfp[test_ind,:],X_test_molchars_std),axis = 1)
    y_train = labels[train_ind]
    y_test = labels[test_ind]
    #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,labels,test_size = .5, shuffle = True, stratify = labels, random_state = 2562)
    bin_y_train, bin_y_test = [1 if x ==2 else x for x in y_train],[1 if x ==2 else x for x in y_test]
    
    #%%
#sgd linear svm
sgd_linear_SVM = sklearn.linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=100000, 
                                                    tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
                                                    validation_fraction=0.1, n_iter_no_change=5, class_weight='balanced', warm_start=False, average=False)
sgd_linear_SVM_model = sgd_linear_SVM.fit(X_train,y_train)
sgd_lSVM_preds = sgd_linear_SVM_model.predict(X_test)
bin_sgd_linear_SVM_model = sgd_linear_SVM.fit(X_train,bin_y_train)
bin_sgd_lSVM_preds = bin_sgd_linear_SVM_model.predict(X_test)