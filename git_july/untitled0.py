#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:37:27 2019

@author: gabriel
"""

'''Make boxplot to compare tuned_1 and ranked epsilon greedy'''



from comet_ml import Experiment
exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                        project_name="iter_plotting", workspace="gdreiman1", disabled = False
                        )
exp.log_code = True
exp.log_other('Hypothesis','''These are my plots combining the GCNN 100 epoch random run with other classifiers also with random selection and 5% ''')
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
data_dir = '/home/gabriel/Dropbox/UCL/Thesis/Data'
first8 = 'tuned_1_svmmod0.pkl'
second8 = 'tuned_1_svmmod1.pkl'
third8 = 'tuned_1_svmmod2.pkl'

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
expr_1 = first8
expr_2 = second8
expr_3 = third8

from iter_plot_help_funcs import find_active_percents,plot_metrics,plot_prec_rec_curve,plot_prec_rec_vs_tresh,get_checkpointsdf
tune_list = []
for exper in [expr_1,expr_2,expr_3]:
    exper = find_active_percents(exper,exp)
    random_checkpoint = get_checkpointsdf(exper,10,5)
    class_selection_list=[]
    for _,row in random_checkpoint.iterrows():
        class_selection_list.append(row.Classifier+'_tuned')
    random_checkpoint['Exp_Cond'] = class_selection_list
    tune_list.append(random_checkpoint)
merged_tune = pd.concat(tune_list)
#%%
'''Now get the ranked_epsilon_greedy'''
first8 = 'ranked_diverse_run_cont0.pkl'
second8 = 'ranked_diverse_run_cont1.pkl'
third8 = 'ranked_diverse_run_cont2.pkl'
first894 = 'ranked_diverse_run0.pkl'
second894 = 'ranked_diverse_run1.pkl'
third894 = 'ranked_diverse_run2.pkl'
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
save_path = os.path.join(data_dir,first894)
pickle_off = open(save_path,'rb')
first894=pickle.load(pickle_off)
pickle_off.close() 
save_path = os.path.join(data_dir,second894)
pickle_off = open(save_path,'rb')
second894=pickle.load(pickle_off)
pickle_off.close() 
save_path = os.path.join(data_dir,third894)
pickle_off = open(save_path,'rb')
third894=pickle.load(pickle_off)
pickle_off.close() 


#combine the 3 dfs with GCNN RF,SVM,LGBM data
expr_1 = pd.concat([first8,first894])
expr_2 = pd.concat([second8,second894])
expr_3 = pd.concat([third8,third894])
from iter_plot_help_funcs import find_active_percents,plot_metrics,plot_prec_rec_curve,plot_prec_rec_vs_tresh,get_checkpointsdf,set_sns_pal
ranked_list = []
for exper in [expr_1,expr_2,expr_3]:
    exper = find_active_percents(exper,exp)
    random_checkpoint = get_checkpointsdf(exper,10,5)
    class_selection_list=[]
    for _,row in random_checkpoint.iterrows():
        class_selection_list.append(row.Classifier+'_untuned')
    random_checkpoint['Exp_Cond'] = class_selection_list
    ranked_list.append(random_checkpoint)
merged_ranked = pd.concat(ranked_list)
merged_df = pd.concat([merged_ranked,merged_tune[merged_tune.Classifier != 'random']])
hue_order_list = list(np.unique(merged_df.Exp_Cond))
hue_order_list.reverse()
set_sns_pal('paired')
g=sns.catplot(x='Classifier',y='Score',hue='Exp_Cond',col='Lib%',data=merged_df,
            kind='box', hue_order=hue_order_list)
plt.suptitle('Percent Actives Recovered at 30% and 50% of library')
plt.show()   
