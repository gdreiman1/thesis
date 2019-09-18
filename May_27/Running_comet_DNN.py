# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:59:43 2019

@author: gdrei
"""

from Comet_DNN import comet_DNN
import os

for AID in ['AID_893','AID_628','AID_894']:
    
    AID_path = os.path.join(r'C:\Users\gdrei\Dropbox\UCL\Thesis\Data', AID) 
    save_path = AID_path+ '\\' + AID +'_processed.pkl'
    comet_DNN(save_path, False)