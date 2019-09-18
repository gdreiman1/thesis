'''Trying to compare both over and under sampling on the non DNN set, arbitrarily chose 1:3 ratio active to inactive'''


def comet_Fold(save_path,embedding_type, model_type, bin_labels):
    from comet_ml import Experiment
    exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                            project_name="80_10_baseline", workspace="gdreiman1", disabled = False
                            )
    exp.log_code = True
    #turn off comet logging comments
    import os    
    #os.environ['COMET_LOGGING_FILE_LEVEL'] = 'WARNING'  
    import warnings
    warnings.filterwarnings('ignore') 
    import pickle
    import pandas as pd
    import numpy as np
    import sklearn as sklearn
    from sklearn.metrics import precision_recall_fscore_support as prf
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import matplotlib.pyplot as plt
    import seaborn as sns
    import ntpath
    from imblearn.over_sampling import RandomOverSampler
    #choosing a 4:1 Inactive to Active ratio
    ros = RandomOverSampler(sampling_strategy= 0.33,random_state=42)
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy= 0.33,random_state=42)

    '''Comet Saving Zone'''
    def comet_addtional_info(exp,save_path, metrics_dict,X_test, y_test,embedding_type, model_type):
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
        model_save = Path(folder+'/'+model_type+'_'+embedding_type+'_'+exp_num+'metrics_dict.pkl')
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
    def train_DNN(X_train,X_test,y_train,y_test,split_ID):
        import tensorflow as tf
        #tf.enable_eager_execution()
#        from keras import backend as K
        from tensorflow.keras.models import Sequential 
        from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras.utils import to_categorical
#        def focal_loss(y_true, y_pred):
#            gamma = 2.0
#            alpha = 0.25
#            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#    #        pt_1 = K.clip(pt_1, 1e-3, .999)
#    #        pt_0 = K.clip(pt_0, 1e-3, .999)
#    
#            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log( pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 ))
        
        #bias for predictions       
        fl_pi = 0.01
        final_bias = -np.log((1-fl_pi)/fl_pi)
        num_labels = len(set(y_test)) 
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
                     batch_size=500,class_weight = class_weights,
                     shuffle=True,
                     verbose=0)
        NN_test_preds = fast_NN.predict(X_test)
        prec,rec,f_1,supp = prf(y_test, np.argmax(NN_test_preds,axis=1), average=None)
        class_rep = sklearn.metrics.classification_report(y_test,np.argmax(NN_test_preds,axis=1))
        exp.log_other('Classification Report'+split_ID,class_rep)
        mcc = sklearn.metrics.matthews_corrcoef(y_test, np.argmax(NN_test_preds,axis=1))
        
        #if first iteration, report model parameters to comet
#        if split_ID == '0':
#            exp.log_parameters(lgbm.get_params())
        return prec,rec,f_1,supp,mcc 
    #from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
   
    def flatten(d, parent_key='', sep='_'): 
        import collections
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    def calc_and_save_metrics(X_train,X_test,y_train,y_test,split_index,model_type,
                              embedding_type,AID,metric_names,metric_dict_list,split_info,split_num,little_split_num):
        '''Takes in test and train data + labels, computes metrics and saves them
        as a dict inside of the provided list. Returns this list.'''
        prec,rec,f_1,supp,mcc = classifier_train(X_train,X_test,y_train,y_test,split_info)
        results_array = np.concatenate((prec,rec,f_1,supp)).tolist()+[mcc]
        if little_split_num == 'NaN':
            split_size = '80%'
        else:
            split_size = '10%'
        metric_dict_list.append(dict(zip(metric_names,[model_type,embedding_type,AID,split_num,little_split_num,split_size,split_index,split_info]+results_array)))
        return metric_dict_list
    '''Begin the actual experiment'''
    #get data cleaned
    pickle_off = open(save_path,'rb')
    activity_table=pickle.load(pickle_off)
    pickle_off.close()
    #get AID
    folder,base = ntpath.split(save_path)
    #split file name at second _ assumes file save in AID_xxx_endinfo.pkl
    AID, _,end_info = base.rpartition('_')
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
    metric_names = ['Classifier','Embedding','AID','80% Split Number','10% Split Number','Train Split Size','ID','Split Info','prec_Inactive','prec_Active','rec_Inactive','rec_Active','f_1_Inactive','f_1_Active','supp_Inactive','supp_Active','mcc']
   
    #determine model type
    classifier_dict = {'SVM': train_SVM, 'RF': train_RF, 'LGBM':train_LGBM,'DNN':train_DNN}
    #set dummy variable to func that trains specified model
    classifier_train = classifier_dict[model_type]
    metric_dict_list = []
    #using labels as a dummy for X
    for split_num, [train_ind, test_ind] in enumerate(big_splitter.split(labels,labels)):
        #indexs which split the data comes from X.X ie big.little
        split_index = str(split_num)
        little_split_num = 'NaN'
        '''Regular Sample'''
        split_info = 'Split'+split_index+' 80% train' + 'BaseRatio'
        #get test/train index
        X_train,X_test,y_train,y_test=get_Scaled_Data(train_ind,test_ind,X_mfp,activity_table,labels,bin_labels)
        #train model and get back classwise metrics
        over_X_train,over_y_train = ros.fit_resample(X_train,y_train)
        metric_dict_list = calc_and_save_metrics(over_X_train,X_test,over_y_train,y_test,split_index,model_type,
                              embedding_type,AID,metric_names,metric_dict_list,split_info,split_num,little_split_num)
        '''Over Sample'''
        split_info = 'Split'+split_index+' 80% train' + 'OverSample'
        #get test/train index
        X_train,X_test,y_train,y_test=get_Scaled_Data(train_ind,test_ind,X_mfp,activity_table,labels,bin_labels)
        #train model and get back classwise metrics
        over_X_train,over_y_train = ros.fit_resample(X_train,y_train)
        metric_dict_list = calc_and_save_metrics(over_X_train,X_test,over_y_train,y_test,split_index,model_type,
                              embedding_type,AID,metric_names,metric_dict_list,split_info,split_num,little_split_num)
        '''Under Sample'''
        split_info = 'Split'+split_index+' 80% train' + 'UnderSample'
        #get test/train index
        X_train,X_test,y_train,y_test=get_Scaled_Data(train_ind,test_ind,X_mfp,activity_table,labels,bin_labels)
        #train model and get back classwise metrics
        under_X_train,under_y_train = rus.fit_resample(X_train,y_train)
        #print('active ratio is:',sum(under_y_train)/len(under_y_train))
        metric_dict_list = calc_and_save_metrics(under_X_train,X_test,under_y_train,y_test,split_index,model_type,
                              embedding_type,AID,metric_names,metric_dict_list,split_info,split_num,little_split_num)
        
        for little_split_num, [little_train_ind, little_test_ind] in enumerate(little_splitter.split(labels[train_ind],labels[train_ind])):
            split_index = str(split_num)+'.'+str(little_split_num)
            '''Regular Sample'''
            split_info = 'Split'+split_index+' 10% train' + 'BaseRatio'
            #get test/train index
            X_train,X_test,y_train,y_test=get_Scaled_Data(train_ind,test_ind,X_mfp,activity_table,labels,bin_labels)
            #train model and get back classwise metrics
            over_X_train,over_y_train = ros.fit_resample(X_train,y_train)
            if len(set(y_train)) == 2:
                metric_dict_list = calc_and_save_metrics(over_X_train,X_test,over_y_train,y_test,split_index,model_type,
                              embedding_type,AID,metric_names,metric_dict_list,split_info,split_num,little_split_num)
            else:
                print('Skipped ' + split_info)
            '''Over Sample'''
            #get test/train index
            X_train,X_test,y_train,y_test=get_Scaled_Data(little_train_ind,test_ind,X_mfp,activity_table,labels,bin_labels)
            over_X_train,over_y_train = ros.fit_resample(X_train,y_train)
            split_info = 'Split' + str(split_num) +' 10% train' +'OverSample'
            #train model and get back classwise metrics
            #check if train_split contains both postive and negative labels
            if len(set(y_train)) == 2:
                metric_dict_list = calc_and_save_metrics(over_X_train,X_test,over_y_train,y_test,split_index,model_type,
                              embedding_type,AID,metric_names,metric_dict_list,split_info,split_num,little_split_num)
            else:
                print('Skipped ' + split_info)
            '''UnderSample'''
            #get test/train index
            X_train,X_test,y_train,y_test=get_Scaled_Data(little_train_ind,test_ind,X_mfp,activity_table,labels,bin_labels)
            under_X_train,under_y_train = rus.fit_resample(X_train,y_train)
            split_info = 'Split' + str(split_num) +' 10% train' +'UnderSample'
            #train model and get back classwise metrics
            #check if train_split contains both postive and negative labels
            if len(set(y_train)) == 2:
                metric_dict_list = calc_and_save_metrics(under_X_train,X_test,under_y_train,y_test,split_index,model_type,
                              embedding_type,AID,metric_names,metric_dict_list,split_info,split_num,little_split_num)
            else:
                print('Skipped ' + split_info)
    # now convert metric_dict_list to df:
    metrics_df = pd.DataFrame(metric_dict_list)
    #set Split_ID to inded
    #now plot all the columns
    #first make a new df column to ID things as either split 
    cols_to_plot = ['prec_Inactive','prec_Active','rec_Inactive','rec_Active','f_1_Inactive','f_1_Active','supp_Inactive','supp_Active','mcc']
    #turn off plotting
    plt.ioff()
    for metric in cols_to_plot:
        #make sns boxplot
        ax = sns.boxplot(x='Split Info', y=metric,  data=metrics_df)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
        plt.tight_layout()
        #log the plot
        exp.log_figure()
        plt.clf()
    ''' now we're going to go through and calculate means and stds for 3 diff groups
        1) the 5 80% train runs
        2) the 5 sets of 8 10% runs
        3) the 40 total 10% runs
        we save each in a list as a pd Series with a name explaining the contents'''
    
    #now add list of dicts of averages to metrics df
    #convert metrics_df to metric dict and log it

    #save metric_df to current folder
    comet_addtional_info(exp,save_path,metrics_df,X_test,y_test,embedding_type, model_type)
    return metrics_df
    
    
import os
import pandas as pd
from pathlib import Path
import sys

#log = open("/media/Data/gabriel/Thesis/Data/log.txt", "w")

count = 0
for AID in ['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']:
    for model in ['DNN']:
        if 'win' in sys.platform:
            AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
        else:
            AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
        save_path = AID_path+ '/' + AID +'_processed.pkl'
#        try:
        if count == 0:
            multiple_metrics_df = comet_Fold(save_path, 'MFPMolChars',model,True)
            count+=1
        else:
            multiple_metrics_df = pd.concat([multiple_metrics_df,comet_Fold(save_path, 'MFPMolChars',model,True)])
#        except Exception as e:
#            print(e)
#            pass
#        try:
        multiple_metrics_df = pd.concat([multiple_metrics_df,comet_Fold(save_path, 'MFP',model,True)])
#        except Exception as e:
#            print(e)
#            pass
#        try:
        multiple_metrics_df = pd.concat([multiple_metrics_df,comet_Fold(save_path, 'MolChars',model,True)])
#        except Exception as e:
#            print(e)
#            pass
#dump as pkl
if 'win' in sys.platform:
    multi_dump_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', 'multiple_metrics_df_over_under.pkl') 
    multiple_metrics_df.to_pickle(multi_dump_path)
else:
    multi_dump_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', 'multiple_metrics_df_over_under_dnn.pkl') 
    multiple_metrics_df.to_pickle(multi_dump_path)