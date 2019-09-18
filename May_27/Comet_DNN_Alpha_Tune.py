# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:39:12 2019

@author: gdrei
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:49:27 2019

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
def comet_DNN(save_path,embedding_type,bin_labels):
    from comet_ml import Experiment
    exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                            project_name="DNN_baseline", workspace="gdreiman1")
    exp.log_code = True
    exp.log_other('Notes','NN_arch same as exp from 7/6 that had good prec/rec, added .1 bias to elu layers, added pi from fl paper, using binary labels')
    import tensorflow as tf
    tf.enable_eager_execution()
    import pickle
    import pandas as pd
    import numpy as np
    import sklearn
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support as prf
   
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
        class_rep = sklearn.metrics.classification_report(y_test,np.argmax(NN_test_preds,axis=1))
        
        #print(class_rep)
        if len(set(y_test)) == 2:
            try:
                prec,rec,f_1,supp = prf(y_test, np.argmax(NN_test_preds,axis=1), average=None)
                single_roc(NN_test_preds,y_test)
            except:
                pass
                    
        else:
            
            try:
                prec,rec,f_1,supp = prf(y_test, np.argmax(NN_test_preds,axis=1), average=None)
                multi_roc(NN_test_preds,to_categorical(y_test),'_',len(set(y_test)))
            except:
                pass
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
        if len(set(y_test)) == 2:
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
        if bin_labels == True:
            label_status = 'binary'
        else:
            label_status = 'multiple'
            
        tags = [AID,end_info,model_type,label_status]
        exp.add_tags(tags)
        exp.add_tag('4_layer')
        exp.add_tag(embedding_type)
        #save ROC curve
        exp.log_figure(figure_name = 'ROC-Pres/Recall',figure=plt)
        plt.show()
        exp.end()

    model_type = 'DNN'
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
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=None, random_state=2562)
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
        bin_y_train, bin_y_test = np.array([1 if x == 0 else 0 for x in y_train]),np.array([1 if x ==0  else 0 for x in y_test])
        if bin_labels==True:
            y_test = bin_y_test
            y_train = bin_y_train
    
    #from https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
    def focal_loss(y_true, y_pred):
        gamma = 2
        alpha = 3
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#        pt_1 = K.clip(pt_1, 1e-3, .999)
#        pt_0 = K.clip(pt_0, 1e-3, .999)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log( pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 ))
    #bias for predictions       
    fl_pi = 0.01
    final_bias = -np.log((1-fl_pi)/fl_pi)
    num_labels = len(set(y_test)) 
    #calculate class weights
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
    tf.keras.backend.clear_session()
    fast_NN = Sequential(name = 'quick')
    #fast_NN.add(GaussianNoise(.5))
    fast_NN.add(Dense(512, activation = 'sigmoid', name = 'input'))
    #fast_NN.add(Dropout(0.5))
    fast_NN.add(Dense(128, activation = 'relu', name = 'first',bias_initializer = tf.keras.initializers.Constant(value=0.1)))
    #fast_NN.add(Dropout(0.5))
    fast_NN.add(Dense(64, activation = 'relu', name = 'second',bias_initializer = tf.keras.initializers.Constant(value=0.1)))
    #fast_NN.add(Dropout(0.5))
    fast_NN.add(Dense(16, activation = 'relu', name = 'third',bias_initializer = tf.keras.initializers.Constant(value=0.1)))
    #fast_NN.add(Dropout(0.25))
    fast_NN.add(Dense(num_labels, activation = 'softmax', name = 'predict',bias_initializer = tf.keras.initializers.Constant(value=final_bias)))
    fast_NN.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics=['categorical_accuracy', tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    fast_NN_model = fast_NN.fit(X_train, to_categorical(y_train),
                 validation_data=(X_test,to_categorical(y_test)),
                 epochs=10,
                 batch_size=500,class_weight=class_weights,
                 shuffle=True,
                 verbose=1)
    comet_addtional_info(exp,fast_NN,save_path,X_test,y_test,embedding_type,model_type)
    
    
import os

for AID in ['AID_624255','AID_1345083','AID_449739','AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']:
    
    AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
    save_path = AID_path+ '\\' + AID +'_processed.pkl'
    comet_DNN(save_path,'MFPMolChars',True )
    comet_DNN(save_path,'MFP', True )    
    comet_DNN(save_path,'MolChars',True )    
    
    
 

