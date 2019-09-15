#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:33:44 2019

@author: gabriel
"""
'''Initaly attempts to use graph cnn for calssification'''
import numpy as np
import tensorboard
import tensorflow as tf
#import mol2graph.py

import deepchem as dc
import pickle,os
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
from imblearn.over_sampling import RandomOverSampler
#choosing a 4:1 Inactive to Active ratio
ros = RandomOverSampler(sampling_strategy= 0.33)

for AID in ['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']:
    AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
    save_path = AID_path+ '/' + AID +'smiles_processed.pkl'
    pickle_off = open(save_path,'rb')
    activity_table=pickle.load(pickle_off)
    pickle_off.close()
    graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    #below has shape np.squeeze(np.array(features), axis=1), valid_inds
    X_embeddings = dc.data.data_loader.featurize_smiles_df(activity_table,featurizer=graph_featurizer, field="PUBCHEM_OPENEYE_CAN_SMILES")
    labels = [1 if x == 'Active' else 0 for x in activity_table['PUBCHEM_ACTIVITY_OUTCOME'] ]
    
    #Now we just store as Numpy Dataset? Or maybe I don't need to do that
    from deepchem.data.datasets import NumpyDataset # import NumpyDataset
    
    dataset = NumpyDataset(np.squeeze(X_oversampled),y_oversampled)
    splitter = dc.splits.splitters.RandomSplitter()
    trainset,testset = splitter.train_test_split(dataset)
    X_oversampled,y_oversampled = ros.fit_resample(np.atleast_2d(X_embeddings[0]).T,labels)
    test_classifier = GraphConvModel(1,graph_conv_layers=[64,64],dense_layer_size=128,dropout = 0.5,model_dir='models',mode='classification',number_atom_features=75, 
                                     n_classes=2, uncertainty=False,use_queue = False,tensorboard=True)
    test_classifier.fit(trainset,nb_epoch=10)
    dnn_preds = test_classifier.predict(testset)
    break
#    hp = dc.molnet.preset_hyper_parameters
#    param = hp.hps[ 'graphconvreg' ]
#    print(param['batch_size'])
#    g = tf.Graph()
#    graph_model = dc.nn.SequentialGraph( 75 )
#    graph_model.add( dc.nn.GraphConv( int(param['n_filters']), 75, activation='relu' ))
#    graph_model.add( dc.nn.BatchNormalization( epsilon=1e-5, mode=1 ))
#    graph_model.add( dc.nn.GraphPool() )
#    graph_model.add( dc.nn.GraphConv( int(param['n_filters']), int(param['n_filters']), activation='relu' ))
#    graph_model.add( dc.nn.BatchNormalization( epsilon=1e-5, mode=1 ))
#    graph_model.add( dc.nn.GraphPool() )
#    graph_model.add( dc.nn.Dense( int(param['n_fully_connected_nodes']), int(param['n_filters']), activation='relu' ))
#    graph_model.add( dc.nn.BatchNormalization( epsilon=1e-5, mode=1 ))
#    #graph_model.add( dc.nn.GraphGather(param['batch_size'], activation='tanh'))
#    graph_model.add( dc.nn.GraphGather( 10 , activation='tanh'))
#    
#    with tf.Session() as sess:
#        model_graphconv = dc.models.MultitaskGraphRegressor( graph_model,
#                                                          1,
#                                                          75,
#                                                         batch_size=10,
#                                                         learning_rate = param['learning_rate'],
#                                                         optimizer_type = 'adam',
#                                                         beta1=.9,beta2=.999)
#        model_graphconv.fit( trainset, nb_epoch=30 )
#    
#    train_scores = {}
#    regression_metric = dc.metrics.Metric( dc.metrics.pearson_r2_score, np.mean )
#    train_scores['graphconvreg'] = model_graphconv.evaluate( trainset,[ regression_metric ]  )
#    p=model_graphconv.predict( testset )
#    '''
#    for i in range( len(p )):
#        print( p[i], testset.y[i] )
#    '''
#    print(train_scores) 