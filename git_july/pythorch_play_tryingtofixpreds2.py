#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 19:23:57 2019

@author: gabriel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:49:26 2019

@author: gabriel
"""

'''Copying blogdef on neuralfp!
https://iwatobipen.wordpress.com/2019/04/05/make-graph-convolution-model-with-geometric-deep-learning-extension-library-for-pytorch-rdkit-chemoinformatics-pytorch/
''' 
#from comet_ml import Experiment
#exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
#                        project_name="iter_baseline", workspace="gdreiman1", disabled = False
#                        )
#exp.log_code = True
#exp.log_other('Hypothesis','want to see bottle necks!!')


import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch,os
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
import mol2graph
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
import pickle
from imblearn.over_sampling import RandomOverSampler
import itertools
import time
#choosing a 4:1 Inactive to Active ratio
ros = RandomOverSampler(sampling_strategy= 0.33)
from sklearn.model_selection import StratifiedShuffleSplit
 
#for AID in ['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']:
for AID in ['AID_1345083']:

    AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
    save_path = AID_path+ '/' + AID +'mol_processed.pkl'
    pickle_off = open(save_path,'rb')
    activity_table=pickle.load(pickle_off)
    pickle_off.close() 
    big_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2562)
    little_splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.1, test_size=0.2, random_state=2562)
    labels = np.array([1 if x =='Active' else 0 for x in activity_table['PUBCHEM_ACTIVITY_OUTCOME']])
    for big_train_ind, big_test_ind in big_splitter.split(activity_table,activity_table['PUBCHEM_ACTIVITY_OUTCOME']):
        train_X = np.atleast_2d(activity_table['MOL'].iloc[big_train_ind]).T
        train_y = labels[big_train_ind]
        test_X = np.atleast_2d(activity_table['MOL'].iloc[big_test_ind]).T
        test_y= labels[big_test_ind]
        train_X_oversampled, train_y_oversampled = ros.fit_resample(train_X,train_y)
        train_X = [mol2graph.mol2vec(m) for m in np.squeeze(train_X_oversampled)]
        test_X = [mol2graph.mol2vec(m) for m in np.squeeze(test_X)]
        #attach train labels to data
        for data, label in zip(train_X,train_y_oversampled):
            data.y = torch.tensor([[label]],dtype=torch.float)
        for data, label in zip(test_X,test_y):
            data.y = torch.tensor([[label]],dtype=torch.float)
        train_loader = DataLoader(train_X, batch_size=128, shuffle=True, drop_last=False,num_workers = 8)
        test_loader = DataLoader(test_X, batch_size=128, shuffle=True, drop_last=False,num_workers = 8)
        train_noshuff = DataLoader(train_X, batch_size=128, shuffle=False, drop_last=False,num_workers = 8)


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
                x = F.dropout(x, p=0.2, training=self.training)
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
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.fc3(x)
                x = F.sigmoid(x)
#                print('x shape:',x.shape)
                return x
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
#                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = F.binary_cross_entropy_with_logits(output, data.y,pos_weight=pos_weight)
#                loss = criterion(output, data.y)
                loss.backward()
                loss_all += loss.item() * data.num_graphs
                optimizer.step()
            return loss_all / len(train_X)
        def test(loader):
            model.eval()
            correct = 0
            pred_list = []
            for data in loader:
                data = data.to(device)
                output = model.test(data)
                pred = output >=0.5
                pred_list.append(pred.flatten().tolist())
#                print(len(pred_list))
                true = data.y >= 0.5
                correct += pred.eq(true).sum().item()
            return correct / len(loader.dataset), list(itertools.chain.from_iterable(pred_list))
        def predictProba(loader):
            model.eval()
            pred_list = []
            for data in loader:
                data = data.to(device)
                output = model.test(data)
                pred = output
                pred_list.append(pred.flatten().tolist())
            return pred_list
        hist = {"loss":[], "acc":[], "test_acc":[]}
        start_time = time.time()

        for epoch in range(1, 2):
            train_loss = train(epoch)
            train_acc,_ = test(train_loader)
            test_acc,_ = test(test_loader)
            hist["loss"].append(train_loss)
            hist["acc"].append(train_acc)
            hist["test_acc"].append(test_acc)
            print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Train_acc: {train_acc:.3}, Test_acc: {test_acc:.3}')
        print('Time for 10 epochs: ',time.time() - start_time)
#        ax = plt.subplot(1,1,1)
#        ax.plot([e for e in range(1,100)], hist["loss"], label="train_loss")
#        ax.plot([e for e in range(1,100)], hist["acc"], label="train_acc")
#        ax.plot([e for e in range(1,100)], hist["test_acc"], label="test_acc")
#        plt.xlabel("epoch")
#        ax.legend()
#        exp.log_figure()
#        break
#    break
#import pandas as pd
#from Iterative_help_funcs import calc_and_save_metrics
#train_preds = predictProba(train_noshuff)
#train_preds =[entry for pred_list in train_preds for entry in pred_list]
#test_metric_list = calc_and_save_metrics(train_y_oversampled,train_preds,'pytorchGCNN','graph',AID,test_metric_list,3,'train')
#test_metric_df = pd.DataFrame(test_metric_list)