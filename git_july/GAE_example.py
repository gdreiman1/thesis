#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:41:57 2019

@author: gabriel
"""

  
import os.path as osp

import argparse,sys,pickle,os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, global_mean_pool,global_max_pool
from torch_geometric.data import DataLoader


torch.manual_seed(12345)


AID_list =['AID_995']

for AID in AID_list:
        if 'win' in sys.platform:
            AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
        else:
            AID_path = os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', AID) 
        save_path = AID_path+ '/' + AID +'graph_processed.pkl'
        pickle_off = open(save_path,'rb')
        activity_table=pickle.load(pickle_off)
        pickle_off.close()

train_loader = DataLoader(activity_table['Graph Rep'], batch_size=128, shuffle=True, drop_last=True,num_workers = 8)
encode_loader =  DataLoader(activity_table['Graph Rep'], batch_size=256, shuffle=False, drop_last=False,num_workers = 8)
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)
        # Map into 2*out_channels dimentions with 


    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


#This is the size of the latent embedding
channels = 32
# We have 75 origional features
num_features = 75
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#dev = torch.device('cpu')

model = GAE(Encoder(num_features, channels).to(dev))
#data.train_mask = data.val_mask = data.test_mask = data.y = None
#data = model.split_edges(data)
#x, train_edge_index = data.x.to(dev), data.edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train(loader):
    model.train()
    loss_all = 0
    for data in loader:
        x, train_edge_index = data.x.to(dev), data.edge_index.to(dev)
        optimizer.zero_grad()
        z = model.encode(x, train_edge_index)
        loss = model.recon_loss(z, train_edge_index)
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs

    return loss_all / len(loader.dataset)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, pos_edge_index)
    return model.test(z, pos_edge_index, pos_edge_index)
def flatten(loader):
    model.eval()
    embedding_list = []
    with torch.no_grad():
        for data in loader:
            z = model.encode(data.x, data.edge_index)
            meanpooled = global_mean_pool(z,data.batch)
            maxpooled = global_max_pool(z,data.batch)
            embeddings = torch.cat((meanpooled,maxpooled),1)
            embedding_list.append(embeddings.tolist())
    return embedding_list
hist = {"loss":[]}
for epoch in range(50, 200):
    train_loss = train(train_loader)
    hist["loss"].append(train_loss)
    print('Epoch: {:03d}, loss: {:.4f}'.format(epoch, train_loss))
    
def flatten_single(data):
    model.eval()
    with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
    return z
            

encode_iter = iter(encode_loader)
data = next(encode_iter)
z_test = flatten_single(data)
meanpooled = global_mean_pool(z_test,data.batch)
