#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:07:55 2019

@author: gabriel
"""
from comet_ml import Experiment
exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                        project_name="iter_plotting", workspace="gdreiman1", disabled = False
                        )
exp.log_code = True
exp.log_other('Hypothesis','''These are my plots from my random gcnn modification ''')
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
data_dir = '/home/gabriel/Dropbox/UCL/Thesis/Data'
first8 = 'gcnn_mod0.pkl'
second8 = 'gcnn_mod1.pkl'
third8 = 'gcnn_mod2.pkl'

save_path = os.path.join(data_dir,first8)
pickle_off = open(save_path,'rb')
first8=pickle.load(pickle_off)
pickle_off.close() 
save_path = os.path.join(data_dir,second8)
pickle_off = open(save_path,'rb')
second8=pickle.load(pickle_off)
pickle_off.close() 
save_path = os.path.join(data_dir,third8)
pickle_off = open(save_path,'rb')
third8=pickle.load(pickle_off)
pickle_off.close() 



#combine the 3 dfs with GCNN RF,SVM,LGBM data
expr_1 = pd.concat([first8,first894])
expr_2 = pd.concat([second8,second894])
expr_3 = pd.concat([third8,third894])

from iter_help_funcs import find_active_percents,plot_metrics,plot_prec_rec_curve,plot_prec_rec_vs_tresh
for exper in [expr_1,expr_2,expr_3]:
    exper = find_active_percents(exper,exp)
    plot_metrics(exper,exp)
#    plot_prec_rec_curve(exper,exp)
#    plot_prec_rec_vs_tresh(exper,exp)
#    break
#get gCNN rows:
df_list = []
for rep,exper in enumerate([first8,second8,third8]):
    exper_gcnn = exper[exper['Classifier']=='GCNN_pytorch']
    
    for _,row in exper_gcnn.iterrows():
        hist = row['hist']
        row_df = pd.DataFrame(hist)
        test = pd.melt(row_df.reset_index(),id_vars=['index'],value_name = 'Score',var_name = 'Metric')
        test['AID']= rep
        test['Iter_num']=row['Iteration Number']
        df_list.append(test)
merged_df = pd.concat(df_list)
g = sns.relplot(x='index',y='Score',hue='Metric',row='AID',col='Iter_num',data=merged_df,kind='line')
g=g.set(ylim=(0,1))
exp.end()
