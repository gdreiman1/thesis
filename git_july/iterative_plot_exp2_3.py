#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:31:15 2019

@author: gabriel
"""

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
exp.log_other('Hypothesis','''These are my plots from the intial iterations "Iter_2" and "Iter_3" ''')
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
data_dir = '/home/gabriel/Dropbox/UCL/Thesis/Data'
random_run = 'first_random_9iter_run.pkl'
diverse_run = 'first_diverse_9iter_run.pkl'

save_path = os.path.join(data_dir,random_run)
pickle_off = open(save_path,'rb')
random_run=pickle.load(pickle_off)
pickle_off.close() 
save_path = os.path.join(data_dir,diverse_run)
pickle_off = open(save_path,'rb')
diverse_run=pickle.load(pickle_off)
pickle_off.close() 

expr_1 = random_run[random_run.AID != 'AID_605']
expr_2 = diverse_run[diverse_run.AID != 'AID_605']
'''This section plots the graphs'''
from iter_plot_help_funcs import find_active_percents,plot_metrics,plot_prec_rec_curve,plot_prec_rec_vs_tresh,plot_avg_percent_found,set_sns_pal
set_sns_pal('unpaired')
for exper in [expr_1,expr_2]:
    exper = find_active_percents(exper,exp)
    plot_metrics(exper,exp)
#    plot_prec_rec_curve(exper,exp)
#    plot_prec_rec_vs_tresh(exper,exp)
#    break
#get gCNN rows:
plot_avg_percent_found(expr_1,'Mean Active Recovery for Classifiers with Diverse Exploration')
plot_avg_percent_found(expr_2,'Mean Active Recovery for Classifiers with Random Exploration \n Initial Selection Strategy',10,5)

'''Check difference between the random and diverse selections'''
from iter_plot_help_funcs import get_checkpointsdf
random_checkpoint = get_checkpointsdf(expr_1,10,5)
class_selection_list=[]
for _,row in random_checkpoint.iterrows():
    class_selection_list.append(row.Classifier+'_random')
random_checkpoint['Exp_Cond'] = class_selection_list
diverse_checkpoint = get_checkpointsdf(expr_2,10,5)
class_selection_list=[]
for _,row in diverse_checkpoint.iterrows():
    class_selection_list.append(row.Classifier+'_diverse')
diverse_checkpoint['Exp_Cond'] = class_selection_list
merged_23 = pd.concat([random_checkpoint,diverse_checkpoint])
hue_order_list = list(np.unique(merged_23.Exp_Cond))
hue_order_list.reverse()
set_sns_pal('paired')
g=sns.catplot(x='Classifier',y='Score',hue='Exp_Cond',col='Lib%',data=merged_23,
            kind='box', hue_order=hue_order_list)
plt.suptitle('Percent Actives Recovered at 30% and 50% of library')
plt.show()   
#,palette = sns.color_palette("Paired")
for exper in [expr_1,expr_2]:
    exper_gcnn = exper[exper['Classifier']=='GCNN_pytorch']
    df_list = []
    for _,row in exper_gcnn.iterrows():
        hist = row['hist']
        row_df = pd.DataFrame(hist)
        test = pd.melt(row_df.reset_index(),id_vars=['index'],value_name = 'Score',var_name = 'Metric')
        test['AID']= row['AID']
        test['Iter_num']=row['Iteration Number']
        df_list.append(test)
    merged_df = pd.concat(df_list)
    g = sns.relplot(x='index',y='Score',hue='Metric',row='AID',col='Iter_num',data=merged_df,kind='line')
    g=g.set(ylim=(0,1))
exp.end()
