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
exp.log_other('Hypothesis','''These are my plots combining the GCNN 100 epoch random run with other classifiers also with random selection and 5% ''')
import pickle
import os
import pandas as pd

data_dir = '/home/gabriel/Dropbox/UCL/Thesis/Data'
firstsection = 'second_greedy_NOTF_100epoch_iter_rerun.pkl'
secondsection = 'second_greedy_NOTF_100epoch_iter_rerun_cont.pkl'
thirdsection = 'second_greedy_NOTF_100epoch_iter_rerun_end.pkl'
tfsection = 'second_greedy_TF_iter_run.pkl'
save_path = os.path.join(data_dir,firstsection)
pickle_off = open(save_path,'rb')
first=pickle.load(pickle_off)
pickle_off.close() 
save_path = os.path.join(data_dir,secondsection)
pickle_off = open(save_path,'rb')
second=pickle.load(pickle_off)
pickle_off.close() 
save_path = os.path.join(data_dir,thirdsection)
pickle_off = open(save_path,'rb')
third=pickle.load(pickle_off)
pickle_off.close() 
save_path = os.path.join(data_dir,tfsection)
pickle_off = open(save_path,'rb')
tf_section=pickle.load(pickle_off)
pickle_off.close() 

#filter out aid_995 from first
first = first[first.AID != 'AID_995']
#filter out AID893
second = second[second.AID != 'AID_893']
#combine the 3 dfs with GCNN RF,SVM,LGBM data
merged_df = pd.concat([first,second,third,tf_section])


#now cycle thru the dnn and merged df and inset the DNN dfs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
found_percent_list = []
scan_percent_list = [] 
for _,row in merged_df.iterrows():
    #if a test row, then calculate percent found
    if row['test_train'] == 'test':
        num_actives_remaining = row['supp_Active']
        num_inactive_remaining = row['supp_Inactive']
        AID = row['AID']
        classifier = row['Classifier']
        iter_num = row['Iteration Number']
        num_actives_in_first_test = merged_df[((merged_df['AID']==AID) & (merged_df['Classifier'] == classifier) 
                                         & (merged_df['Iteration Number'] == 0) & (merged_df['test_train'] == 'test'))]['supp_Active']
        num_inactives_in_first_test = merged_df[((merged_df['AID']==AID) & (merged_df['Classifier'] == classifier) 
                                         & (merged_df['Iteration Number'] == 0) & (merged_df['test_train'] == 'test'))]['supp_Inactive']
        found_percent_list.append(1.0-float(num_actives_remaining/num_actives_in_first_test))
        # divide remaining members of test pool by  num in in first test pool/0.9 to account for 10% initial train set
        scan_percent_list.append(1.0-float((num_actives_remaining+num_inactive_remaining)/((num_actives_in_first_test+num_inactives_in_first_test)/0.9)))
    #if train, append an 'nan'
    else:
        found_percent_list.append(np.nan)
        scan_percent_list.append(np.nan)

merged_df['Percent Active Found']=found_percent_list
merged_df['Percent_lib_scanned']=scan_percent_list
from iter_plot_help_funcs import plot_avg_percent_found_vs_scanned
plot_avg_percent_found_vs_scanned(merged_df,'Actives Recovery vs % Library Scanned for \n Greedy Experiment',10,5)
g = sns.lineplot(x="Percent_lib_scanned", y="Score", hue='Classifier', data=active_find_plot[active_find_plotMetric=='Percent Active F0und'.],kind='line',legend='full',markers= True)

#%%

#now calculate what %
percent_pred_active_list = []
for _,row in merged_df.iterrows():
    #if a test row, then calculate percent found
    if row['test_train'] == 'test':
        [tn, fp], [fn, tp] = row['conf_matrix']
        num_pred_active = (tp + fp)
        num_actives_remaining = row['supp_Active']
        AID = row['AID']
        classifier = row['Classifier']
        iter_num = row['Iteration Number']
        batch_size = np.sum(np.sum(merged_df[((merged_df['AID']==AID) & (merged_df['Classifier'] == classifier) 
                                         & (merged_df['Iteration Number'] == 0) & (merged_df['test_train'] == 'train'))]['conf_matrix']))
        percent_pred_active_list.append(num_pred_active/batch_size)
    #if train, append an 'nan'
    else:
        percent_pred_active_list.append(np.nan)
merged_df['Percent Batch Pred Active']=percent_pred_active_list
#sns.lineplot(x="Iteration Number", y="Percent Batch Pred Active",
#             hue="Classifier", data=merged_df.dropna())
#exp.log_figure()
#
#plt.show()
#plt.clf()

melted = merged_df.melt(id_vars = ['AID','Classifier','Iteration Number','Embedding','test_train','Percent_lib_scanned'],var_name = 'Metric',value_name='Score')
mcc_auc_plot = melted[(melted['Metric'].isin(['auc','mcc']))&
                    (melted['test_train']=='test')]
active_find_plot = melted[(melted['Metric'].isin(['Percent Active Found','Percent Batch Pred Active']))&
                    (melted['test_train']=='test')]
prec_rec_plot = melted[(melted['Metric'].isin(['prec_Active','rec_Active']))&
                    (melted['test_train']=='test')]
mcc_auc_plot.Score = mcc_auc_plot.Score.astype(float)
active_find_plot.Score = active_find_plot.Score.astype(float)
prec_rec_plot.Score = prec_rec_plot.Score.astype(float)

#%%
'''Plot 3 figures: AUC/MCC,%found/totalfound,recall/precision @0.5'''
g = sns.relplot(x="Iteration Number", y="Score", hue='Classifier',style="Metric", col="AID", col_wrap=3, data=mcc_auc_plot,kind='line',legend='full',markers= True,ci = None )
exp.log_figure()
g = sns.relplot(x="Iteration Number", y="Score", hue='Classifier',style="Metric", col="AID", col_wrap=3, data=active_find_plot,kind='line',legend='full',markers= True,ci = None )
exp.log_figure()
g = sns.relplot(x="Iteration Number", y="Score", hue='Classifier',style="Metric", col="AID", col_wrap=3, data=prec_rec_plot,kind='line',legend='full',markers= True,ci = None )
exp.log_figure()
g = sns.relplot(x="Percent_lib_scanned", y="Score", hue='Classifier',style="Metric", col="AID", col_wrap=3, data=active_find_plot,kind='line',legend='full',markers= True,ci = None )
exp.log_figure()
'''Plot prec/recall curves for all points '''
df = merged_df[merged_df['test_train']=='test']
# Initialize the figure
fig, axs = plt.subplots(9, 10)
plt.style.use('seaborn-darkgrid')

# create an aid dict
AID_list =['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_596','AID_893','AID_894']

 
# create a color palette
from cycler import cycler
from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette())
colors = cmap.colors
custom_cycler = (cycler(color=colors[:5]) 
                 + cycler(lw=[1, 1, 1, 1, 1]))
#set color pallettes for all of these
for ax in axs.flatten().tolist():
    ax.set_prop_cycle(custom_cycler)
# multiple line plot
num=0
for _,row in df.iterrows():
    ax_row = AID_list.index(row['AID'])
    ax_col = int(row['Iteration Number'])
    #get prec rec from df row
    rec = row['rec_array'].tolist()
    prec = row['prec_array'].tolist()
    recs, precs = zip(*sorted(zip(rec, prec)))
    #sort the recall so that we can view it on y axis
    axs[ax_row,ax_col].plot(precs,recs,label=row['Classifier'])
    # Find the right spot on the plot

 
# general title
plt.suptitle("Recall Precision curves for classifiers, iterations, AIDs", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)
plt.ylabel('Recall')
plt.xlabel('Precision')
# add legends
#for ax in axs.flatten().tolist():
#    ax.legend()
 
# Axis title
for index, ax in enumerate(axs[0]):
    ax.set_title('Iteration '+str(index))

for index, ax in enumerate(axs[:,0]):
    ax.set_ylabel(AID_list[index], rotation=90, size='large')
handles, labels = axs[-1,-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.tight_layout()
plt.show()

#%%
'''Plot both prec and recall vs threshold'''
df = merged_df[merged_df['test_train']=='test']
# Initialize the figure
fig, axs = plt.subplots(9, 10, figsize= (11.6, 8.2))
plt.style.use('seaborn-darkgrid')
fig.suptitle("Recall Precision curves for classifiers, iterations, AIDs", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)
#plt.ylabel('Recall/Precision',labelpad=20)
#plt.xlabel('Threshold')
# create an aid dict
AID_list =['AID_1345083','AID_624255','AID_449739','AID_995','AID_938','AID_628','AID_596','AID_893','AID_894']

 
# create a color palette
from cycler import cycler
from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette())
colors = cmap.colors
custom_cycler = (cycler(color=[a for sublist in zip(colors[:5],colors[:5]) for a in sublist]) 
                 + cycler(lw=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) +
                 cycler(linestyle=['-', '--','-', '--','-', '--','-', '--','-', '--']))
#set color pallettes for all of these
for ax in axs.flatten().tolist():
    ax.set_prop_cycle(custom_cycler)
# multiple line plot
for _,row in df.iterrows():
    ax_row = AID_list.index(row['AID'])
    ax_col = int(row['Iteration Number'])
    #get prec rec from df row
    rec = row['rec_array'].tolist()
    prec = row['prec_array'].tolist()
    thresh = row['thresh_array'].tolist()
    axs[ax_row,ax_col].plot(thresh,prec[:-1],label=row['Classifier'] + ' Precision')
    axs[ax_row,ax_col].plot(thresh,rec[:-1],label=row['Classifier'] + ' Recall')

    # Find the right spot on the plot

 
# general title
#fig.suptitle("Recall Precision curves for classifiers, iterations, AIDs", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)
#plt.ylabel('Recall/Precision',labelpad=20)
#plt.xlabel('Threshold')

# Axis title
for index, ax in enumerate(axs[0]):
    ax.set_title('Iteration '+str(index))

for index, ax in enumerate(axs[:,0]):
    ax.set_ylabel(AID_list[index], rotation=90, size='large')
handles, labels = axs[-1,-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
exp.end()
