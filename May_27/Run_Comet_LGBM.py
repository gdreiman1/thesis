# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:04:03 2019

@author: gdrei
"""

from light_gbm_start import comet_lgbm as lgb_com
name_stump = r'C:\Users\gdrei\Dropbox\UCL\Thesis\May_13'
proc_list = ['\AID_628_processed.pkl','\AID_449739_processed.pkl', 
             '\AID_894_processed.pkl','\AID_893_processed.pkl',
             '\AID_624255_processed.pkl', '\AID_1345083_processed.pkl']
for name in proc_list:
    lgb_com(name_stump+name)
