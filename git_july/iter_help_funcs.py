#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:15:15 2019

@author: gabriel
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def find_active_percents(merged_df,exp):
    found_percent_list = []
    for _,row in merged_df.iterrows():
        #if a test row, then calculate percent found
        if row['test_train'] == 'test':
            num_actives_remaining = row['supp_Active']
            AID = row['AID']
            classifier = row['Classifier']
            iter_num = row['Iteration Number']
            num_actives_in_first_test = merged_df[((merged_df['AID']==AID) & (merged_df['Classifier'] == classifier) 
                                             & (merged_df['Iteration Number'] == 0) & (merged_df['test_train'] == 'test'))]['supp_Active']
            found_percent_list.append(1.0-float(num_actives_remaining/num_actives_in_first_test))
        #if train, append an 'nan'
        else:
            found_percent_list.append(np.nan)
    merged_df['Percent Active Found']=found_percent_list
    
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
    return merged_df

def plot_metrics(merged_df,exp):
    melted = merged_df.melt(id_vars = ['AID','Classifier','Iteration Number','Embedding','test_train'],var_name = 'Metric',value_name='Score')
    mcc_auc_plot = melted[(melted['Metric'].isin(['auc','mcc']))&
                        (melted['test_train']=='test')]
    active_find_plot = melted[(melted['Metric'].isin(['Percent Active Found','Percent Batch Pred Active']))&
                        (melted['test_train']=='test')]
    prec_rec_plot = melted[(melted['Metric'].isin(['prec_Active','rec_Active']))&
                        (melted['test_train']=='test')]
    mcc_auc_plot.Score = mcc_auc_plot.Score.astype(float)
    active_find_plot.Score = active_find_plot.Score.astype(float)
    prec_rec_plot.Score = prec_rec_plot.Score.astype(float)
    '''Plot 3 figures: AUC/MCC,%found/totalfound,recall/precision @0.5'''
    g = sns.relplot(x="Iteration Number", y="Score", hue='Classifier',style="Metric", col="AID", col_wrap=3, data=mcc_auc_plot,kind='line',legend='full',markers= True,ci = None )
    exp.log_figure()
    g = sns.relplot(x="Iteration Number", y="Score", hue='Classifier',style="Metric", col="AID", col_wrap=3, data=active_find_plot,kind='line',legend='full',markers= True,ci = None )
    exp.log_figure()
    g = sns.relplot(x="Iteration Number", y="Score", hue='Classifier',style="Metric", col="AID", col_wrap=3, data=prec_rec_plot,kind='line',legend='full',markers= True,ci = None )
    exp.log_figure()
def plot_prec_rec_curve(merged_df,exp): 
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

def plot_prec_rec_vs_tresh(merged_df,exp):
    '''Plot both prec and recall vs threshold'''
    df = merged_df[merged_df['test_train']=='test']
    # Initialize the figure
    fig, axs = plt.subplots(9, 10, figsize= (11.6, 8.2))
    plt.style.use('seaborn-darkgrid')
    plt.suptitle("Recall Precision curves for classifiers, iterations, AIDs", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)
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
    plt.legend(handles, labels, loc='upper right')
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.imshow(aspect='auto')