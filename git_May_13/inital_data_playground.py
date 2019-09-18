# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:52:18 2019

@author: gdrei

This is just playing around with a random HTS from pubchem
https://pubchem.ncbi.nlm.nih.gov/bioassay/1345083
http://moreisdifferent.com/2017/9/21/DIY-Drug-Discovery-using-molecular-fingerprints-and-machine-learning-for-solubility-prediction/#basic-concepts-of-fingerprinting

"""
#Imports

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
'''Read in our results'''
expr_table = pd.read_csv(r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_1345083_datatable.csv')
path_to_file = r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\1668495245601928809.sdf\1668495245601928809.sdf'
# this deals out SD formatted molecules
suppl = Chem.SDMolSupplier(path_to_file)
#make a df to hold fingerprint, activity, and ID
activity = pd.DataFrame(expr_table.loc[5:,'PUBCHEM_ACTIVITY_OUTCOME'], index=expr_table.loc[5:,'PUBCHEM_CID'])
#fp_activity.join(expr_table.loc[:,'PUBCHEM_ACTIVITY_OUTCOME'])
#%%
# check consistancy
def Consistant_Checker(expr_table):
    #remove nans

    nan_inds = expr_table.index[expr_table['PUBCHEM_CID'].isna()].tolist()
    expr_table.drop(nan_inds,inplace = True)
    #find duplicated cids, removed those that have inconsistent results, return thoe removed cids
    from collections import Counter
    duplicated_cid = [k for k,v in Counter(expr_table.loc[:,'PUBCHEM_CID']).items() if v>1]
    dropped = []
    for x in duplicated_cid:
        #get locations of duplicated CID
        dup_indxs = expr_table.index[expr_table['PUBCHEM_CID']==x].tolist()
        #boolean where results beeing all the same returns 1
        if len(set(expr_table.loc[dup_indxs, 'PUBCHEM_ACTIVITY_OUTCOME'])):
            #drop repated experiemnts after first instance
            expr_table.drop(dup_indxs[1:],inplace = True)
        else:
            #its inconsistent, drop everything
            expr_table.drop(dup_indxs,inplace = True)
            dropped.append(x)
    expr_table['PUBCHEM_CID'] =expr_table['PUBCHEM_CID'].astype(int).astype(str)
    return(expr_table,dropped)

def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))
#get morgen fps:
def get_mfp_hash(fp_length,suppl):
    '''Takes desired fp length and supply, returns df with cols 'PUBCHEM_CID' 
    and "MFP'''
    fp_list = [[mol.GetProp('_Name'),ExplicitBitVect_to_NumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits = fp_length))] for mol in suppl]
    return pd.DataFrame(fp_list, columns=['PUBCHEM_CID','MFP'])
#align the activiy with the fps
def Align_with_Activity(activity_table,char_list, removed):
    '''Takes a dataframe "activity_table" which holds info about activity and a 
    list "cahr_list" of characteristics for each molecule that form part of
    the embedding, and aligns the two, returning a new df'''
    #get CID's from acitivty_table to sort cahr_list with 
    srt = {b: i for i, b in enumerate(expr_table.loc[:,'PUBCHEM_CID'])}
    
    '''remove cids from char_list if we removed them from our activity table
    (they are removed in Consistant_Checker if they have inconsitant activities
    on the same CID)'''
    #go thru outer list, getting char rows. If the cid in char_row[0] is not in removed (list of cids that were removed)
    char_list[:] = [char_row for char_row in char_list if char_row[0] not in removed]
    
    #get the intersection of the CIDs in activity table and char_list
    intersected_CIDs = list[set(expr_table['PUBCHEM_CID']) & set(char_list[:,0])]
    
            
    #when same length its mfps so creat header 
    if len(activity_table) == len(char_list):
        name_vect = ['name','MFP']
        to_sort = char_list
        sorted_chars = np.array(sorted(to_sort, key=lambda x: srt[x[0]]))
        activity_table['MFP'] = sorted_chars[:,1]
    else:
        name_vect = char_list[0]
        to_sort = char_list[1:]
        sorted_chars = np.array(sorted(to_sort, key=lambda x: srt[x[0]]))
        for i,char_name in enumerate(name_vect[1:],1):
            activity_table[char_name] = sorted_chars[:,i]
    return(activity_table)
def Add_Mol_Chars(suppl):
    char_names = ['Chi0','Chi0n','Chi0v','Chi1',\
    'Chi1n','Chi1v','Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v',\
    'EState_VSA1','EState_VSA10','EState_VSA11','EState_VSA2','EState_VSA3',\
    'EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8',\
    'EState_VSA9','FractionCSP3','HallKierAlpha','HeavyAtomCount','Ipc',\
    'Kappa1','Kappa2','Kappa3','LabuteASA','MolLogP','MolMR','MolWt',\
    'NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles',\
    'NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles',\
    'NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms',\
    'NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles',\
    'NumSaturatedRings','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12',\
    'PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5',\
    'PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','RingCount','SMR_VSA1',\
    'SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6','SMR_VSA7',\
    'SMR_VSA8','SMR_VSA9','SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12',\
    'SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7',\
    'SlogP_VSA8','SlogP_VSA9','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',\
    'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',\
    'VSA_EState8','VSA_EState9']
    calc = MolecularDescriptorCalculator(char_names)
    full_list = []

    for mol in suppl:
        if mol is None: continue
        #row_list = [mol.GetProp('_Name')]
        full_list.append([mol.GetProp('_Name')]+list(calc.CalcDescriptors(mol)))
    list_version = [['PUBCHEM_CID']+char_names]+(full_list)
    return(pd.DataFrame(list_version[1:], columns = list_version[0]))

def Read_Process_Save(expr_loc,path_to_sdf,save_path):

    expr_table = pd.read_csv(expr_loc)
    
    # this deals out SD formatted molecules
    suppl = Chem.SDMolSupplier(path_to_sdf)
    #remove repeats
    expr_table,dropped_cids = Consistant_Checker(expr_table.loc[:,'PUBCHEM_CID':'PUBCHEM_ACTIVITY_OUTCOME'])
    #get mfps
    fp_length = 1024
    fp_list = get_mfp_hash(fp_length,suppl)
    #get mol_chars
    mol_chars = Add_Mol_Chars(suppl)    
    #combine the labels and the finger prints
    activity_table = expr_table.merge(fp_list, on='PUBCHEM_CID')
    #combine the labels and the mol_chars
    activity_table = activity_table.merge(mol_chars, on='PUBCHEM_CID')
    activity_table = Align_with_Activity(activity_table,mol_chars,dropped_cids)
    pickle_on = open(save_path,'wb')
    pickle.dump(activity_table,pickle_on)
    pickle_on.close()
path_lists = [[r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_449739_datatable.csv',
 r'C:\Users\gdrei\Downloads\1441395869637736289.sdf\1441395869637736289.sdf',
 r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13\AID_449739_processed.pkl']]
for experiment in path_lists:
    [expr_loc,path_to_sdf,save_path] = experiment[:]
    expr_table = pd.read_csv(expr_loc)
    print(save_path)
    # this deals out SD formatted molecules
    suppl = Chem.SDMolSupplier(path_to_sdf)
    #remove repeats
    expr_table,dropped_cids = Consistant_Checker(expr_table.loc[:,'PUBCHEM_CID':'PUBCHEM_ACTIVITY_OUTCOME'])
    #get mfps
    fp_length = 1024
    fp_list = get_mfp_hash(fp_length,suppl)
    #get mol_chars
    mol_chars = Add_Mol_Chars(suppl)
    #combine the labels and the finger prints
    activity_table = expr_table.merge(fp_list, on='PUBCHEM_CID')
    #combine the labels and the mol_chars
    activity_table = activity_table.merge(mol_chars, on='PUBCHEM_CID')
    pickle_on = open(save_path,'wb')
    pickle.dump(activity_table,pickle_on)
    pickle_on.close()

#%%
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

#standardize the molchars for X and X test


'''model farm'''
#%%
#rand forests
rf = RandomForestClassifier(n_estimators=100, random_state=2562, class_weight="balanced_subsample", n_jobs = -1)
rand_for = rf.fit(X_train,y_train)
rf_preds = rand_for.predict(X_test)
bin_rand_for = rf.fit(X_train,bin_y_train)
bin_rf_preds = bin_rand_for.predict(X_test)
#%%
#adaboost
adaboost = sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R', random_state=2562)
adaboost_model = adaboost.fit(X_train,y_train)
adaboost_preds = adaboost_model.predict(X_test)
bin_adaboost_model = adaboost.fit(X_train,bin_y_train)
bin_adaboost_preds = bin_adaboost_model.predict(X_test)
#%%
#sgd linear svm
sgd_linear_SVM = sklearn.linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=100000, 
                                                    tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
                                                    validation_fraction=0.1, n_iter_no_change=5, class_weight='balanced', warm_start=False, average=False)
sgd_linear_SVM_model = sgd_linear_SVM.fit(X_train,y_train)
sgd_lSVM_preds = sgd_linear_SVM_model.predict(X_test)
bin_sgd_linear_SVM_model = sgd_linear_SVM.fit(X_train,bin_y_train)
bin_sgd_lSVM_preds = bin_sgd_linear_SVM_model.predict(X_test)
#%%
#simple neural net
import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import to_categorical
#from https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score


metrics = Metrics()
 
metrics = Metrics()
fast_NN = Sequential(name = 'quick')
fast_NN.add(Dense(512, input_shape = (len(X_test[0,:]),), activation = 'sigmoid', name = 'input'))
fast_NN.add(Dense(128, activation = 'relu', name = 'first'))
fast_NN.add(Dense(64, activation = 'relu', name = 'second'))
fast_NN.add(Dense(16, activation = 'relu', name = 'third'))
fast_NN.add(Dense(3, activation = 'softmax', name = 'predict'))
fast_NN.compile(loss = [focal_loss], optimizer='adam',metrics=['categorical_accuracy'])
fast_NN_model = fast_NN.fit(X_train, to_categorical(y_train),
             validation_data=(X_test, to_categorical(y_test)),
             epochs=30,
             batch_size=500,
             shuffle=True,
             verbose=False)
bin_fast_NN = Sequential(name = 'bin_quick')
bin_fast_NN.add(Dense(512, input_shape = (len(X_test[0,:]),), activation = 'sigmoid', name = 'input'))
bin_fast_NN.add(Dense(128, activation = 'relu', name = 'first'))
bin_fast_NN.add(Dense(64, activation = 'relu', name = 'second'))
bin_fast_NN.add(Dense(16, activation = 'relu', name = 'third'))
bin_fast_NN.add(Dense(2, activation = 'softmax', name = 'predict'))
bin_fast_NN.compile(loss = [focal_loss], optimizer='adam',metrics=['categorical_accuracy'])
bin_fast_NN_model = bin_fast_NN.fit(X_train, to_categorical(bin_y_train),
             validation_data=(X_test, to_categorical(bin_y_test)),
             epochs=60,
             batch_size=1000,
             shuffle=True,
             verbose=True)
'''below is a modified version of the binary now using a sigmoid predictor'''
bin_fast_NN = Sequential(name = 'bin_quick')
bin_fast_NN.add(Dense(512, input_shape = (len(X_test[0,:]),), activation = 'sigmoid', name = 'input'))
bin_fast_NN.add(Dropout(0.7))
bin_fast_NN.add(Dense(128, activation = 'relu', name = 'first'))
bin_fast_NN.add(Dropout(0.7))
bin_fast_NN.add(Dense(64, activation = 'relu', name = 'second'))
bin_fast_NN.add(Dropout(0.7))
bin_fast_NN.add(Dense(16, activation = 'relu', name = 'third'))
bin_fast_NN.add(Dropout(0.5))
bin_fast_NN.add(Dense(1, activation = 'sigmoid', name = 'predict'))
bin_fast_NN.compile(loss = [focal_loss], optimizer='adam',metrics=['binary_accuracy'])
bin_fast_NN_model = bin_fast_NN.fit(X_train, np.array(bin_y_train),
             validation_data=(X_test, np.array(bin_y_test)),
             epochs=60,
             batch_size=1000,
             shuffle=True,
             verbose=True)
#plotROC/AUC
NN_test_preds = fast_NN.predict(X_test)
multi_roc(NN_test_preds,to_categorical(y_test),'NN_Test',3)


bin_NN_test_preds = bin_fast_NN.predict(X_test)
multi_roc(bin_NN_test_preds,to_categorical(bin_y_test),'bin_NN_Test',2)
#%%
'''this takes a really long time with a big training set, but might be managable
 when we start doing iterative screening'''
#RBF SVM
svm_rbf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True,
              probability=False, tol=0.001, cache_size=200, class_weight="balanced", 
             verbose=False, max_iter=1000, decision_function_shape='ovr', random_state=None)
SVM_rbf_model = svm_rbf.fit(X_train[:10000,:],y_train[:10000])
rbf_SVM_preds = svm_rbf.predict(X_test)

#%%
print('Random Forests')
print(sklearn.metrics.classification_report(y_test,rf_preds))
print('bin_Random Forests')
print(sklearn.metrics.classification_report(bin_y_test,bin_rf_preds))
print('adaboost_model')
print(sklearn.metrics.classification_report(y_test,adaboost_preds))
print('bin_adaboost_model')
print(sklearn.metrics.classification_report(bin_y_test,bin_adaboost_preds))
print('SGD lSVM')
print(sklearn.metrics.classification_report(y_test,sgd_lSVM_preds))
print('bin_SGD lSVM')
print(sklearn.metrics.classification_report(bin_y_test,bin_sgd_lSVM_preds))
print('NN')
print(sklearn.metrics.classification_report(y_test,np.argmax(NN_test_preds, axis = 1)))
print('bin_NN')
print(sklearn.metrics.classification_report(bin_y_test,np.argmax(bin_NN_test_preds, axis = 1)))
#print('RBF SVM')
#print(sklearn.metrics.classification_report(y_test,rbf_SVM_preds))
le.inverse_transform([0,1,2])





#%%
#helper funcs
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

def nn_train_plot(history):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
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
    plt.show()
    
    
