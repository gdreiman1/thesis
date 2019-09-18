# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:01:40 2019

@author: gdrei
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:10:02 2019

@author: gdrei
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:46:45 2019

@author: gdrei
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:23:47 2019

@author: gdrei

Light GBM operation

"""


#from comet_ml import Experiment
#exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
#                        project_name="baseline", workspace="gdreiman1")
#exp.log_code = True
import tensorflow as tf
tf.enable_eager_execution()
import pickle
import pandas as pd
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.linear_model import SGDClassifier

def train_DNN(X_train,X_test,y_train,y_test,y_labels):
        import tensorflow as tf
        tf.enable_eager_execution()
        from keras import backend as K
        from tensorflow.keras.models import Sequential 
        from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras.utils import to_categorical
        def focal_loss(y_true, y_pred):
            gamma = 2.0
            alpha = 0.25
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    #        pt_1 = K.clip(pt_1, 1e-3, .999)
    #        pt_0 = K.clip(pt_0, 1e-3, .999)
    
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log( pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 ))
        
        #bias for predictions       
        fl_pi = 0.01
        final_bias = -np.log((1-fl_pi)/fl_pi)
        num_labels = len(set(y_test)) 
        from sklearn.utils import class_weight
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_labels),
                                                 y_labels)
        tf.keras.backend.clear_session()
        fast_NN = Sequential(name = 'quick')
        fast_NN.add(GaussianNoise(.5))
        fast_NN.add(Dense(512, activation = 'sigmoid', name = 'input'))
        fast_NN.add(Dropout(0.5))
        fast_NN.add(Dense(128, activation = 'relu', name = 'first',bias_initializer = tf.keras.initializers.Constant(value=0.1)))
        fast_NN.add(Dropout(0.5))
        fast_NN.add(Dense(64, activation = 'relu', name = 'second',bias_initializer = tf.keras.initializers.Constant(value=0.1)))
        fast_NN.add(Dropout(0.5))
        fast_NN.add(Dense(16, activation = 'relu', name = 'third',bias_initializer = tf.keras.initializers.Constant(value=0.1)))
        fast_NN.add(Dropout(0.25))
        fast_NN.add(Dense(1, activation = 'linear', name = 'predict',bias_initializer = tf.keras.initializers.Constant(value=final_bias)))
        fast_NN.compile(loss = 'mean_squared_error', optimizer='adam')
        from imblearn.keras import BalancedBatchGenerator
        from imblearn.over_sampling import RandomOverSampler
#        training_generator = BalancedBatchGenerator(X_train, y_train, sampler=RandomOverSampler(0.5), batch_size=64, random_state=42)
        fast_NN_model = fast_NN.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=100,batch_size=1024,shuffle=True,   class_weight=class_weights,                                  verbose=1)
        NN_test_preds = fast_NN.predict(X_test)
        prec,rec,f_1,supp = prf(y_test, np.argmax(NN_test_preds,axis=1), average=None)
        class_rep = sklearn.metrics.classification_report(y_test,np.argmax(NN_test_preds,axis=1))
        mcc = sklearn.metrics.matthews_corrcoef(y_test, np.argmax(NN_test_preds,axis=1))
        
        #if first iteration, report model parameters to comet
#        if split_ID == '0':
#            exp.log_parameters(lgbm.get_params())
        return NN_test_preds 
from sklearn.preprocessing import StandardScaler, LabelEncoder
'''Comet Saving Zone'''
def comet_SVM(save_path,bin_labels):
    model_type = 'SVM_regress'
    #get data cleaned
    pickle_off = open(save_path,'rb')
    activity_table=pickle.load(pickle_off)
    pickle_off.close()
    #get length of MFP
    fp_length = len(activity_table.iloc[5]['MFP'])
    #simple neural net
    
    scaler = StandardScaler(copy = False)
    le = LabelEncoder()
    labels = le.fit_transform(activity_table['PUBCHEM_ACTIVITY_OUTCOME'])
    score = np.array(activity_table['PUBCHEM_ACTIVITY_SCORE'])
    #split data:
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.1, random_state=2562)
    X_mfp = np.concatenate(np.array(activity_table['MFP'])).ravel()
    X_mfp = X_mfp.reshape((-1,fp_length))
    for train_ind, test_ind in splitter.split(X_mfp,labels):
        #get start and end index for molchars
        MC_start = activity_table.columns.get_loc('Chi0')
        #need to add 1 bc exclusive indexing
        MC_end = activity_table.columns.get_loc('VSA_EState9')+1
        # standardize data
        X_train_molchars_std = scaler.fit_transform(np.array(activity_table.iloc[train_ind,MC_start:MC_end]))
        X_test_molchars_std = scaler.transform(np.array(activity_table.iloc[test_ind,MC_start:MC_end]))
        X_train = np.concatenate((X_mfp[train_ind,:],X_train_molchars_std),axis = 1)
        X_test = np.concatenate((X_mfp[test_ind,:],X_test_molchars_std),axis = 1)
        y_train_score = score[train_ind]
        y_test_score = score[test_ind]
        y_train_labels = labels[train_ind]
        #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,labels,test_size = .5, shuffle = True, stratify = labels, random_state = 2562)
        #remapping active to 1 and everything else to zero
        bin_y_train, bin_y_test = np.array([1 if x == 0 else 0 for x in y_train]),np.array([1 if x ==0  else 0 for x in y_test])
        if bin_labels==True:
                y_test = bin_y_test
                y_train = bin_y_train
        
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import SGDRegressor
    sgd_linear_SVM = SGDRegressor(loss='epsilon_insensitive', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=100000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)
    sgd_linear_SVM_model = sgd_linear_SVM.fit(X_train,y_train_score)
    dnn_preds = train_DNN(X_train,X_test,y_train_score,y_test_score,y_train_labels)
    sgd_lSVM_preds = sgd_linear_SVM_model.predict(X_test)
    class_preds = [1 if x >= 40 else 0 for x in sgd_lSVM_preds]
    dnn_class_preds = [1 if x >= 10 else 0 for x in dnn_preds]

    print(sklearn.metrics.classification_report(y_test,dnn_class_preds))
    zeroed_preds = [ 0 if x <=0 else 1 if x >= 100 else x/100 for x in sgd_lSVM_preds]

    #comet_addtional_info(exp,sgd_linear_SVM_model,save_path,X_test,y_test)



import os

for AID in ['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']:

    AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
    save_path = AID_path+ '\\' + AID +'_processed.pkl'
    bin_labels = True
    break
    comet_SVM(save_path, True)
    