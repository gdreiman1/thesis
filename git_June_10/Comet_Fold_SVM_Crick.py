# -*- coding: utf-8 -*-


def comet_Fold(save_path,embedding_type, model_type, bin_labels):
    from comet_ml import Experiment
    exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                            project_name="80_10_baseline", workspace="gdreiman1")
    exp.log_code = True
    import pickle
    import numpy as np
    import sklearn as sklearn
    from sklearn.metrics import precision_recall_fscore_support as prf
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    '''Comet Saving Zone'''
    def comet_addtional_info(exp,save_path, metrics_dict,X_test, y_test,embedding_type, model_type):
        #get AID number
        import ntpath
        #get base file name
        folder,base = ntpath.split(save_path)
        #split file name at second _ assumes file save in AID_xxx_endinfo.pkl
        AID, _,end_info = base.rpartition('_')
        exp.add_tag(AID)
        #save data location, AID info, and version info
        exp.log_dataset_info(name = AID, version = end_info, path = save_path)
        #save some informatvie tags:
        tags = [AID,end_info,model_type]
        exp.add_tags(tags)
        exp.add_tag(embedding_type)
        #save metrics_dict in data_folder with comet experiement number associated
        exp_num = exp.get_key()
        model_save = folder+'/'+model_type+'_'+exp_num+'metrics_dict.pkl'
        pickle_on = open(model_save,'wb')
        pickle.dump(metrics_dict,pickle_on)
        pickle_on.close()
        #log trained model location
        exp.log_other('Metrics Dict Path',model_save)
        #tell comet that the experiement is over
        exp.end()
    def get_Scaled_Data(train_ind,test_ind,X_mfp,activity_table,labels,bin_labels):
        #get start and end index for molchars
        MC_start = activity_table.columns.get_loc('Chi0')
        #need to add 1 bc exclusive indexing
        MC_end = activity_table.columns.get_loc('VSA_EState9')+1
        # standardize data    
        scaler = StandardScaler(copy = False)
        #return requested datatype
        if embedding_type == 'MFPMolChars':
            X_train_molchars_std = scaler.fit_transform(np.array(activity_table.iloc[train_ind,MC_start:MC_end]))
            X_test_molchars_std = scaler.transform(np.array(activity_table.iloc[test_ind,MC_start:MC_end]))
            X_train = np.concatenate((X_mfp[train_ind,:],X_train_molchars_std),axis = 1)
            X_test = np.concatenate((X_mfp[test_ind,:],X_test_molchars_std),axis = 1)
        elif embedding_type == 'MFP':
            X_train = X_mfp[train_ind,:]
            X_test = X_mfp[test_ind,:]
        elif embedding_type == 'MolChars':
            X_train_molchars_std = scaler.fit_transform(np.array(activity_table.iloc[train_ind,MC_start:MC_end]))
            X_test_molchars_std = scaler.transform(np.array(activity_table.iloc[test_ind,MC_start:MC_end]))
            X_train = X_train_molchars_std
            X_test = X_test_molchars_std
        y_train = labels[train_ind]
        y_test = labels[test_ind]
        #remapping active to 1 and everything else to zero
        bin_y_train, bin_y_test = np.array([1 if x == 0 else 0 for x in y_train]),np.array([1 if x ==0  else 0 for x in y_test])
        if bin_labels==True:
            y_test = bin_y_test
            y_train = bin_y_train
        return X_train,X_test,y_train,y_test
    def train_SVM(X_train,X_test,y_train,y_test,split_ID):
        sgd_linear_SVM = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=500000, 
                                                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
                                                        validation_fraction=0.1, n_iter_no_change=5, class_weight='balanced', warm_start=False, average=False)
        sgd_linear_SVM_model = sgd_linear_SVM.fit(X_train,y_train)
        
        sgd_lSVM_preds = sgd_linear_SVM_model.predict(X_test)
        prec,rec,f_1,supp = prf(y_test, sgd_lSVM_preds, average=None)
        class_rep = sklearn.metrics.classification_report(y_test,sgd_lSVM_preds)
        exp.log_other('Classification Report'+split_ID,class_rep)
        mcc = sklearn.metrics.matthews_corrcoef(y_test, sgd_lSVM_preds)
        
        #if first iteration, report model parameters to comet
        if split_ID == '0':
            exp.log_parameters(sgd_linear_SVM_model.get_params())
        return prec,rec,f_1,supp,mcc
    def train_RF(X_train,X_test,y_train,y_test,split_ID):
            
        rf = RandomForestClassifier(n_estimators=100, random_state=2562, class_weight="balanced_subsample", n_jobs = -1)
        rand_for = rf.fit(X_train,y_train)
        rf_preds = rand_for.predict(X_test)
        prec,rec,f_1,supp = prf(y_test, rf_preds, average=None)
        class_rep = sklearn.metrics.classification_report(y_test,rf_preds)
        exp.log_other('Classification Report'+split_ID,class_rep)
        mcc = sklearn.metrics.matthews_corrcoef(y_test, rf_preds)
        
        #if first iteration, report model parameters to comet
        if split_ID == '0':
            exp.log_parameters(rand_for.get_params())
        return prec,rec,f_1,supp,mcc 
    def train_LGBM(X_train,X_test,y_train,y_test,split_ID):
        import lightgbm as lgb
        #make model class
        lgbm_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=500, subsample_for_bin=200000, 
                                        objective='binary', is_unbalance=True, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
                                        subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, 
                                        importance_type='split')
        #train model
        lgbm = lgbm_model.fit(X_train,y_train)
        lgbm_preds = lgbm.predict(X_test)
        prec,rec,f_1,supp = prf(y_test, lgbm_preds, average=None)
        class_rep = sklearn.metrics.classification_report(y_test,lgbm_preds)
        exp.log_other('Classification Report'+split_ID,class_rep)
        mcc = sklearn.metrics.matthews_corrcoef(y_test, lgbm_preds)
        
        #if first iteration, report model parameters to comet
        if split_ID == '0':
            exp.log_parameters(lgbm.get_params())
        return prec,rec,f_1,supp,mcc 
    #get data cleaned
    pickle_off = open(save_path,'rb')
    activity_table=pickle.load(pickle_off)
    pickle_off.close()
    #get length of MFP
    fp_length = len(activity_table.iloc[5]['MFP'])
    #reshape mfp
    X_mfp = np.concatenate(np.array(activity_table['MFP'])).ravel()
    X_mfp = X_mfp.reshape((-1,fp_length))
    le = LabelEncoder()
    labels = le.fit_transform(activity_table['PUBCHEM_ACTIVITY_OUTCOME'])
    #split data:
    from sklearn.model_selection import StratifiedShuffleSplit
    #this is outer 5fold cross validation i.e. 80/20 split
    big_splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2562)
    #inner replicateing the start with 10% of data (or 12.5% of 80% intial split)
    little_splitter = StratifiedShuffleSplit(n_splits=8, test_size = 0.2, train_size=0.125, random_state=2562)
    #this holds all the metrics values that will be stored in comet
    metric_dict = {}
    metric_names = ['prec_Inactive','prec_Active','rec_Inactive','rec_Active','f_1_Inactive','f_1_Active','supp_Inactive','supp_Active','mcc']
    
    #determine model type
    classifier_dict = {'SVM': train_SVM, 'RF': train_RF, 'LGBM':train_LGBM}
    classifier_train = classifier_dict[model_type]
    #using labels as a dummy for X
    for split_num, [train_ind, test_ind] in enumerate(big_splitter.split(labels,labels)):
        #indexs which split the data comes from X.X ie big.little
        split_index = str(split_num)
        #get test/train index
        X_train,X_test,y_train,y_test=get_Scaled_Data(train_ind,test_ind,X_mfp,activity_table,labels,bin_labels)
        #train model and get back classwise metrics
        prec,rec,f_1,supp,mcc = classifier_train(X_train,X_test,y_train,y_test,split_index)
        #add split_index to metric names this assumes 0 = inactive 1 = active!!
        metric_nameandsplit = [x+'_'+split_index for x in metric_names]
        metric_dict.update(zip(metric_nameandsplit,(np.concatenate((prec,rec,f_1,supp)))+[mcc]))
  
        for little_split_num, [little_train_ind, little_test_ind] in enumerate(little_splitter.split(labels[train_ind],labels[train_ind])):
            split_index = str(split_num)+'.'+str(little_split_num)
            #get test/train index
            X_train,X_test,y_train,y_test=get_Scaled_Data(little_train_ind,test_ind,X_mfp,activity_table,labels,bin_labels)
            #train model and get back classwise metrics
            #check if train_split contains both postive and negative labels
            if len(set(y_train)) == 2:
                prec,rec,f_1,supp,mcc = classifier_train(X_train,X_test,y_train,y_test,split_index)
                metric_nameandsplit = [x+'_'+split_index for x in metric_names]
                metric_dict.update(zip(metric_nameandsplit,(np.concatenate((prec,rec,f_1,supp)))+[mcc]))
            else:
                metric_dict[split_index] = 'Split Contained only 1 class, no training'
    
    # = rand_for.predict(X_test)
    exp.log_metrics(metric_dict)
    
    comet_addtional_info(exp,save_path,metric_dict,X_test,y_test,embedding_type, model_type)
    
    
    
import os

for AID in ['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']:
    
    AID_path = os.path.join('/media/Data/gabriel/Thesis/Data/', AID) 
    save_path = AID_path+ '/' + AID +'_processed.pkl'
    comet_Fold(save_path, 'MFPMolChars','LGBM',True)
    comet_Fold(save_path, 'MFP','LGBM',True)
    comet_Fold(save_path, 'MolChars','LGBM',True)