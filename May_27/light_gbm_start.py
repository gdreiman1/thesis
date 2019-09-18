# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:23:47 2019

@author: gdrei

Light GBM operation

"""
def comet_lgbm(save_path):
    from comet_ml import Experiment
    exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                            project_name="baseline", workspace="gdreiman1")
    exp.log_code = True
    
    import pickle
    import pandas as pd
    import lightgbm as lgb
    import numpy as np
    import sklearn
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support as prf
    #%%
    def single_roc(y_preds,y_true):
        
        from sklearn.metrics import roc_curve, auc,precision_recall_curve
        fpr, tpr, _ = roc_curve(y_true, y_preds)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
        plt.plot(recall, precision, color='blue',
                 lw=lw, label='Precision vs Recall')
        # show the plot
        plt.legend(loc="lower right")
        plt.show()
    def multi_roc(y_preds,y_true,name,n_classes):
        import collections
        nested_dict = lambda: collections.defaultdict(nested_dict)
        data_store = nested_dict()
        from sklearn.metrics import roc_curve, auc
        from scipy import interp
        from itertools import cycle
        lw = 2
        name_store = ['Active', 'Inactive', 'Inconclusive']
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true[:, i].ravel(), y_preds[:, i].ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area
        
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of '+ name_store[i]+'(area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Multi-class ROC for '+name+' Split= '+str(count+1))
        plt.title('Multi-class ROC for '+name)
    
        plt.legend(loc="lower right")
        #plt.show()
    #%%
    #save_path = r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_1345083_processed.pkl'
    model_type = 'lgbm'
    #get data cleaned
    pickle_off = open(save_path,'rb')
    activity_table=pickle.load(pickle_off)
    pickle_off.close()
    #get length of MFP
    fp_length = len(activity_table.iloc[5]['MFP'])
    
    
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler(copy = False)
    le = LabelEncoder()
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
        
    #do light gbm
        
    #need to make a lib svm file
    train_data = lgb.Dataset(X_train,label=y_train)
    test_data = lgb.Dataset(X_test,label=y_test)
    #make model class
    lgbm_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=500, subsample_for_bin=200000, 
                                    objective='binary', is_unbalance=True, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
                                    subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, 
                                    importance_type='split')
    #train model
    trained_mod = lgbm_model.fit(X_train,y_train)
    #predict classes and class_probs
    test_class_preds = lgbm_model.predict(X_test)
    test_prob_preds = lgbm_model.predict_proba(X_test)
    #calculate Class report
    class_rep = sklearn.metrics.classification_report(y_test,test_class_preds)
    
    print(class_rep)
    if len(set(y_test)) == 2:
        single_roc(test_prob_preds[:,1],y_test)
        prec,rec,f_1,supp = prf(y_test, test_class_preds, average=None)
    else:
        from tensorflow.keras.utils import to_categorical
        multi_roc(test_prob_preds,to_categorical(y_test),'',3)
        prec,rec,f_1,supp = prf(y_test, test_class_preds, average=None)
    
    
     #%% 
    '''Comet Saving Zone'''
    #get AID number
    import ntpath
    #get base file name
    folder,base = ntpath.split(save_path)
    #split file name at second _ assumes file save in AID_xxx_endinfo.pkl
    AID, _,end_info = base.rpartition('_')
    #save data location, AID info, and version info
    exp.log_dataset_info(name = AID, version = end_info, path = save_path)
    #save model params
    exp.log_parameters(trained_mod.get_params())
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
    exp_num = exp.get_key()
    model_save = folder+'\\'+model_type+'_'+exp_num+'.pkl'
    pickle_on = open(model_save,'wb')
    pickle.dump(trained_mod,pickle_on)
    pickle_on.close()
    #log trained model location
    exp.log_other('Trained Model Path',model_save)
    #save some informatvie tags:
    tags = [AID,end_info,model_type]
    exp.add_tags(tags)
    #save ROC curve
    exp.log_figure(figure_name = 'ROC-Pres/Recall',figure=plt)
    plt.show()

    #tell comet that the experiement is over
    exp.end()
