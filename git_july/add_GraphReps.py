#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:38:58 2019

@author: gabriel
"""
'''Adding the graph embeddings to the big boy pickle file with all chemical info in it'''
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from Iterative_help_funcs import getGraphX
AID_list =['AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']
for AID in ['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_605','AID_596','AID_893','AID_894']:
    getGraphX(AID)