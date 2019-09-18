# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:52:29 2019

@author: gdrei
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:41:40 2019

@author: gdrei
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:28:21 2019

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

def comet_SVM(save_path,embedding_type):
    from comet_ml import Experiment
    exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                            project_name="baseline", workspace="gdreiman1")
    exp.log_code = True
    import tensorflow as tf
    tf.enable_eager_execution()
    import pickle
    import pandas as pd
    import numpy as np
    import sklearn as sklearn
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support as prf
    from sklearn.linear_model import SGDClassifier
    from keras import backend as K
    from tensorflow.keras.models import Sequential 
    
    from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.utils import to_categorical
    
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from ROC_funcs import single_roc,multi_roc
    '''Comet Saving Zone'''
    def comet_addtional_info(exp, model,save_path, X_test, y_test,embedding_type, model_type):
        from tensorflow.keras.utils import to_categorical
        NN_test_preds = model.predict(X_test)
        class_rep = sklearn.metrics.classification_report(y_test,NN_test_preds)
        
        #print(class_rep)
        if len(set(y_test)) == 2:
            prec,rec,f_1,supp = prf(y_test, NN_test_preds, average=None)    
        else:
            prec,rec,f_1,supp = prf(y_test, NN_test_preds, average=None)
      
        #get AID number
        import ntpath
        #get base file name
        folder,base = ntpath.split(save_path)
        #split file name at second _ assumes file save in AID_xxx_endinfo.pkl
        AID, _,end_info = base.rpartition('_')
        exp.add_tag(AID)
        #save data location, AID info, and version info
        exp.log_dataset_info(name = AID, version = end_info, path = save_path)
        #save model params
        #exp.log_parameters(trained_mod.get_params())
        #save metrics report to comet
        if len(f_1) == 2:
            for i,name in enumerate(['Active','Inactive']):
                exp.log_metric('f1 class '+name, f_1[i])
                exp.log_metric('Recall class'+name,rec[i])
                exp.log_metric('Precision class'+name, prec[i])
        else:
            for i,name in enumerate(['Active','Inconclusive','Inactive']):
                exp.log_metric('f1 class '+str(i), f_1[i])
                exp.log_metric('Recall class'+str(i),rec[i])
                exp.log_metric('Precision class'+str(i), prec[i])
            #exp.log_metric('f1 class '+str(i), f_1[i])
            #exp.log_metric('Recall class'+str(i),rec[i])
            #exp.log_metric('Precision class'+str(i), prec[i])
        exp.log_other('Classification Report',class_rep)
         #save model in data_folder with comet experiement number associated
#        exp_num = exp.get_key()
#        model_save = folder+'\\'+model_type+'_'+exp_num+'.pkl'
#        pickle_on = open(model_save,'wb')
#        pickle.dump(fast_NN,pickle_on)
#        pickle_on.close()
#        #log trained model location
#        exp.log_other('Trained Model Path',model_save)
        #save some informatvie tags:
        tags = [AID,end_info,model_type]
        exp.add_tags(tags)
        exp.add_tag('SVM')
        exp.add_tag(embedding_type)
        #save ROC curve
        exp.log_figure(figure_name = 'ROC-Pres/Recall',figure=plt)
        plt.show()
        
        #tell comet that the experiement is over
        exp.end()
  
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
    #split data:
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, train_size=None, random_state=2562)
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
        if embedding_type == 'MFPMolChars':
            X_train = np.concatenate((X_mfp[train_ind,:],X_train_molchars_std),axis = 1)
            X_test = np.concatenate((X_mfp[test_ind,:],X_test_molchars_std),axis = 1)
        elif embedding_type == 'MFP':
            X_train = X_mfp[train_ind,:]
            X_test = X_mfp[test_ind,:]
        elif embedding_type == 'MolChars':
            X_train = X_train_molchars_std
            X_test = X_test_molchars_std
        y_train = labels[train_ind]
        y_test = labels[test_ind]
        #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,labels,test_size = .5, shuffle = True, stratify = labels, random_state = 2562)
        #remapping active to 1 and everything else to zero
       # bin_y_train, bin_y_test = np.array([1 if x == 0 else 0 for x in y_train]),np.array([1 if x ==0  else 0 for x in y_test])
#        if bin_labels==True:
#            y_test = bin_y_test
#            y_train = bin_y_train
    model_type = 'LGBM'
    import lightgbm as lgb
    #make model class
    lgbm_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=500, subsample_for_bin=200000, 
                                    objective='binary', is_unbalance=True, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
                                    subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, 
                                    importance_type='split')
    #train model
    trained_mod = lgbm_model.fit(X_train,y_train)
    
    comet_addtional_info(exp,trained_mod,save_path,X_test,y_test,embedding_type, model_type)
    
    
    
import os

for AID in ['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']:
    
    AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
    save_path = AID_path+ '\\' + AID +'_processed.pkl'
    comet_SVM(save_path, 'MFP')
    comet_SVM(save_path, 'MolChars')