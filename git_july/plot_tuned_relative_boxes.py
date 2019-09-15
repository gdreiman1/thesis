#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:47:11 2019

@author: gabriel
"""
'''Make the final graph: all hyper parameters tuned'''
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
first8 = 'tuned_7_svmmod0.pkl'
second8 = 'tuned_7_svmmod1.pkl'
third8 = 'tuned_7_svmmod2.pkl'
from iter_plot_help_funcs import find_active_percents,plot_metrics,plot_prec_rec_curve,plot_prec_rec_vs_tresh,get_checkpoint35,set_sns_pal

def get_35_tune(pathlist,sizes,expr_num):
    #get experiment info
    first8,second8,third8 = pathlist
    start_size, iter_size = sizes
    explore_cond_list = ['diverse','random']
    #yes I'm using mod2 to index, sue me.
    explore_type = explore_cond_list[expr_num%2]
    exp_cond = 'Exp_'+str(expr_num)+ '  Start:'+str(start_size/100)+', Iter:'+str(iter_size/100)+', Explore:'+explore_type
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
    
    tune_list = []
    for exper in [expr_1,expr_2,expr_3]:
        exper = find_active_percents(exper,exp)
        random_checkpoint = get_checkpoint35(exper,start_size,iter_size)
        class_selection_list=[]
        expr_num_list = []
        for _,row in random_checkpoint.iterrows():
            class_selection_list.append(exp_cond)        
            expr_num_list.append('Exp_'+str(expr_num))

        random_checkpoint['Exp_Cond'] = class_selection_list
        random_checkpoint['Exp_num'] = expr_num_list
        tune_list.append(random_checkpoint)
    merged_tune = pd.concat(tune_list)
    return merged_tune

#set up info lists
pathlist_list = []
for i in range(1,7):
    if i == 6:
        i+=1
    pathlist=[]
    for j in range(3):
        pathlist.append('tuned_'+str(i)+'_svmmod'+str(j)+'.pkl')
    pathlist_list.append(pathlist)
size_list = [[10,5],[15,5],[15,5],[15,10],[15,10],[5,5]]

df35_list =[]
for i in range(6):
    pathlist= pathlist_list[i]
    sizes = size_list[i]
    expr_num = i+1
    if expr_num == 6:
        expr_num +=1
    curr_df = get_35_tune(pathlist,sizes,expr_num)
    df35_list.append(curr_df)
merged_df = pd.concat(df35_list)
g=sns.catplot(x='Exp_num',y='Score',hue='Exp_Cond',col='Classifier',palette=sns.color_palette("Set2", 6),col_wrap=3,data=merged_df,
            kind='box',legend_out=False)
plt.show()

g=sns.catplot(x='Classifier',y='Score',hue='Classifier',col='Exp_num',palette=sns.color_palette("Set2", 5),col_wrap=3,data=merged_df,
            kind='box',legend_out=False)