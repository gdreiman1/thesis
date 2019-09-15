#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:31:23 2019

@author: gabriel
"""

'''Iterative helper funcs'''
#%%
'''Data Scaler'''
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
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
    elif embedding_type == 'Graph':
        '''Getting graph reps from mol2graph that I stored in df'''
        X_train = activity_table['Graph Rep'].iloc[train_ind].tolist()
        X_test = activity_table['Graph Rep'].iloc[test_ind].tolist()
        y_train = activity_table['PUBCHEM_ACTIVITY_OUTCOME'].iloc[train_ind]
        y_test = activity_table['PUBCHEM_ACTIVITY_OUTCOME'].iloc[test_ind]
#       remapping active to 1 and everything else to zero
        bin_y_train, bin_y_test = np.array([1 if x == 'Active' else 0 for x in y_train]),np.array([1 if x =='Active'  else 0 for x in y_test])
        for data, label in zip(X_train,bin_y_train):
            data.y = torch.tensor([[label]],dtype=torch.float)
        for data, label in zip(X_test,bin_y_test):
            data.y = torch.tensor([[label]],dtype=torch.float)

    y_train = activity_table['PUBCHEM_ACTIVITY_OUTCOME'].iloc[train_ind]
    y_test = activity_table['PUBCHEM_ACTIVITY_OUTCOME'].iloc[test_ind]
    #remapping active to 1 and everything else to zero
    bin_y_train, bin_y_test = np.array([1 if x == 'Active' else 0 for x in y_train]),np.array([1 if x =='Active'  else 0 for x in y_test])
    if bin_labels==True:
        y_test = bin_y_test
        y_train = bin_y_train
    return X_train,X_test,y_train,y_test
'''Get molecular graph reps'''
import pickle
import mol2graph

def getGraphX(AID):
    '''Takes in AID, finds graphreps for pytroch implementation'''
    AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
    save_path = AID_path+ '/' + AID +'mol_processed.pkl'
    pickle_off = open(save_path,'rb')
    activity_table=pickle.load(pickle_off)
    pickle_off.close() 
    graph_rep_list = [mol2graph.mol2vec(m) for m in activity_table['MOL']]
    AID_and_graph_rep = pd.DataFrame()
    AID_and_graph_rep['Graph Rep']= graph_rep_list
    AID_and_graph_rep['PUBCHEM_CID'] = activity_table['PUBCHEM_CID']
    
    main_aid_save_path = AID_path+ '/' + AID +'_processed.pkl'
    pickle_off = open(main_aid_save_path,'rb')
    main_activity_table=pickle.load(pickle_off)
    pickle_off.close()
    main_activity_table = main_activity_table.merge(AID_and_graph_rep,on='PUBCHEM_CID')
    save_df = True
    new_aid_save_path = main_aid_save_path = AID_path+ '/' + AID +'graph_processed.pkl'

    if save_df == True:
        main_activity_table.to_pickle(new_aid_save_path)
    return main_activity_table

#%%
    '''Classifier Section'''
    '''SVM'''
def sigmoid(x):
    return 1/(1+np.exp(x))
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def train_SVM(X_train,X_test,y_train,y_test,base_X_train):
    sgd_linear_SVM = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=500000, 
                                                    tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
                                                    validation_fraction=0.1, n_iter_no_change=5, class_weight='balanced', warm_start=False, average=False)
    sgd_linear_SVM_model = sgd_linear_SVM.fit(X_train,y_train)
#    cv_model = CalibratedClassifierCV(sgd_linear_SVM_model,'sigmoid','prefit')
#    cv_model.fit(X_train,y_train)
#    test_sgd_lSVM_preds = cv_model.predict_proba(X_test)
#    train_sgd_lSVM_preds = cv_model.predict_proba(X_train)
#    base_train_sgd_lSVM_preds = cv_model.predict_proba(base_X_train)
#    return train_sgd_lSVM_preds[:,1],test_sgd_lSVM_preds[:,1],base_train_sgd_lSVM_preds[:,1]

    test_sgd_lSVM_preds = sigmoid(sgd_linear_SVM_model.decision_function(X_test))
    train_sgd_lSVM_preds = sigmoid(sgd_linear_SVM_model.decision_function(X_train))
    base_train_sgd_lSVM_preds = sigmoid(sgd_linear_SVM_model.decision_function(base_X_train))
    return train_sgd_lSVM_preds,test_sgd_lSVM_preds,base_train_sgd_lSVM_preds

'''Random Forest'''
def train_RF(X_train,X_test,y_train,y_test,base_X_train):
        
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs = -1)
    rand_for = rf.fit(X_train,y_train)
    test_rf_preds = rand_for.predict_proba(X_test)
    train_rf_preds = rand_for.predict_proba(X_train)
    base_train_rf_preds = rand_for.predict_proba(base_X_train)

    return train_rf_preds[:,1],test_rf_preds[:,1],base_train_rf_preds[:,1]
'''LGBM'''
import lightgbm as lgb
def train_LGBM(X_train,X_test,y_train,y_test,base_X_train):
    #make model class
    lgbm_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=500, subsample_for_bin=200000, 
                                    objective='binary', is_unbalance=True, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
                                    subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, 
                                    importance_type='split')
    #train model
    lgbm = lgbm_model.fit(X_train,y_train)
    test_lgbm_preds = lgbm.predict_proba(X_test)
    train_lgbm_preds = lgbm.predict_proba(X_train)
    base_train_lgbm_preds = lgbm.predict_proba(base_X_train)

    return train_lgbm_preds[:,1],test_lgbm_preds[:,1],base_train_lgbm_preds[:,1]
'''Standard DNN'''
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
from tensorflow.keras.utils import to_categorical
def train_DNN(X_train,X_test,y_train,y_test,base_X_train):

       
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
        fast_NN.add(Dropout(0.5))
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
        test_NN_test_preds = fast_NN.predict(X_test)
        train_NN_test_preds = fast_NN.predict(X_train)
        base_train_NN_test_preds = fast_NN.predict(base_X_train)

        return train_NN_test_preds[:,1],test_NN_test_preds[:,1],base_train_NN_test_preds[:,1],fast_NN_model.history
   
'''Pytorch Graph CNN'''
import torch,os
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
#from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
#from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
#from torch_scatter import scatter_mean
#import mol2graph
#import itertools
def train_PyTorchGCNN(X_train,X_test,y_train,y_test,base_X_train):
    for data, label in zip(X_train,y_train):
        data.y = torch.tensor([[label]],dtype=torch.float)
    for data, label in zip(X_test,y_test):
        data.y = torch.tensor([[label]],dtype=torch.float)
    train_loader = DataLoader(X_train, batch_size=128, shuffle=True, drop_last=True,num_workers = 8)
    test_loader = DataLoader(X_test, batch_size=128, shuffle=True, drop_last=False,num_workers = 8)
#    base_train_loader = DataLoader(base_X_train, batch_size=128, shuffle=True, drop_last=False,num_workers = 8)
    train_loader_noshuff = DataLoader(X_train, batch_size=128, shuffle=False, drop_last=False,num_workers = 8)
    test_loader_noshuff = DataLoader(X_test, batch_size=128, shuffle=False, drop_last=False,num_workers = 8)
    base_train_loader_noshuff = DataLoader(base_X_train, batch_size=128, shuffle=False, drop_last=False,num_workers = 8)
    n_features = 75
    # definenet
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(n_features, 128, cached=False) # if you defined cache=True, the shape of batch must be same!
            self.bn1 = BatchNorm1d(128)
            self.conv2 = GCNConv(128, 64, cached=False)
            self.bn2 = BatchNorm1d(64)
            self.fc1 = Linear(64, 64)
            self.bn3 = BatchNorm1d(64)
            self.fc2 = Linear(64, 64)
            self.fc3 = Linear(64, 1)
             
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = self.bn1(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.bn2(x)
            x = global_add_pool(x, data.batch)
            x = F.relu(self.fc1(x))
            x = self.bn3(x)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=0.4, training=self.training)
            x = self.fc3(x)
#                x = F.sigmoid(x)
#                print('x shape:',x.shape)
            return x
        def test(self,data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = self.bn1(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.bn2(x)
            x = global_add_pool(x, data.batch)
            x = F.relu(self.fc1(x))
            x = self.bn3(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
#            x = torch.sigmoid(x)
#                print('x shape:',x.shape)
            return x
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    def train(epoch):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model.forward(data)
#                print('Output:',output)
#                print('Target:',data.y)
#                print('Output Shape:',output.shape)
#                print('Target Shape:',np.shape(data.y))
#                print('Data Shape:',np.shape(data))
#                print(loss_all)
            pos_weight = 3*torch.ones([1])  # All weights are equal to 1
            pos_weight = pos_weight.to(device)

#            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = F.binary_cross_entropy_with_logits(output, data.y,pos_weight=pos_weight)
#            loss = criterion(output, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(X_train)
    def test(loader):
        model.eval()
        correct = 0
        pred_list = []
        loss_all = 0
        for data in loader:
            data = data.to(device)
            output = model.test(data)
            pred = torch.sigmoid(output) >=0.5
            pred_list.append(pred.flatten().tolist())
            #print(len(pred_list))
            true = data.y >= 0.5
            correct += pred.eq(true).sum().item()
            #calculate loss            
            pos_weight = 3*torch.ones([1])  # All weights are equal to 1
            pos_weight = pos_weight.to(device)
            loss = F.binary_cross_entropy_with_logits(output, data.y,pos_weight=pos_weight)
            loss_all += loss.item() * data.num_graphs
        return correct / len(loader.dataset), loss_all / len(X_test)
    def predictProba(loader):
        model.eval()
        pred_list = []
        for data in loader:
            data = data.to(device)
            output = model.test(data)
            pred = torch.sigmoid(output)
            pred_list.append(pred.flatten().tolist())
        return pred_list
    hist = {"loss":[],"test_loss":[], "acc":[], "test_acc":[]}
    for epoch in range(1, 20):
        train_loss = train(epoch)
        train_acc,_ = test(train_loader)
        test_acc,test_loss = test(test_loader)
        hist["loss"].append(train_loss)
        hist["test_loss"].append(test_loss)
        hist["acc"].append(train_acc)
        hist["test_acc"].append(test_acc)
        #print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Train_acc: {train_acc:.3}, Test_acc: {test_acc:.3}')
    train_preds = predictProba(train_loader_noshuff)
    train_preds = [entry for pred_list in train_preds for entry in pred_list]
    test_preds = predictProba(test_loader_noshuff)
    test_preds = [entry for pred_list in test_preds for entry in pred_list]
    base_train_preds = predictProba(base_train_loader_noshuff)
    base_train_preds = [entry for pred_list in base_train_preds for entry in pred_list]

    return train_preds,test_preds,base_train_preds,hist

'''Random classifier just gets random samples between 0 and 1 for each entry in
    the X array passed to it'''
def train_random_classifier(X_train,X_test,y_train,y_test,base_X_train):
    train_preds = np.random.random(len(y_train))
    test_preds = np.random.random(len(y_test))
    base_train_preds = np.random.random(max(np.shape(base_X_train)))
    return train_preds,test_preds,base_train_preds
#%%
'''Metrics Calculations'''
from sklearn.metrics import precision_recall_curve, auc,confusion_matrix,matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support as prf

def calc_and_save_metrics(y_true,pred_probs,model_type,
                      embedding_type,AID,metric_dict_list,iter_num,test_train,hist):
    '''Takes in test and train data + labels, computes metrics and saves them
    as a dict inside of the provided list. Returns this list.'''
    #save the hist from our DNNS if needed
    history = hist
    
    #make categorical preds at .5 threshold
    #need to grab the DNN's class 1 preds bc doing the categorical embedding
    class_preds = [x>=0.5 for x in pred_probs]
    #calculate all metrics
    prec,rec,f_1,supp = prf(y_true, class_preds, average=None)
    mcc = matthews_corrcoef(y_true, class_preds)
    conf_mat = confusion_matrix(y_true, class_preds)
    prec_array,recall_array,thresh_array = precision_recall_curve(y_true,pred_probs)
    auc_PR = auc(recall_array,prec_array)
    
    results_array = np.concatenate((prec,rec,f_1,supp)).tolist()+[mcc,prec_array,recall_array,thresh_array,conf_mat,auc_PR]
    metric_names = ['Classifier','Embedding','AID','Iteration Number','test_train',
                        'prec_Inactive','prec_Active','rec_Inactive','rec_Active','f_1_Inactive','f_1_Active',
                        'supp_Inactive','supp_Active','mcc','prec_array','rec_array','thresh_array',
                        'conf_matrix','auc','hist']
    metric_dict_list.append(dict(zip(metric_names,[model_type,embedding_type,AID,iter_num,test_train]+results_array+[history])))
    return metric_dict_list