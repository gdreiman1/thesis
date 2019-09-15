#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:15:15 2019

@author: gabriel
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def find_active_percents(merged_df,exp):
    found_percent_list = []
    for _,row in merged_df.iterrows():
        #if a test row, then calculate percent found
        if row['test_train'] == 'test':
            num_actives_remaining = row['supp_Active']
            AID = row['AID']
            classifier = row['Classifier']
            iter_num = row['Iteration Number']
            base_train_row = merged_df[((merged_df['AID']==AID) & (merged_df['Classifier'] == classifier) 
                                         & (merged_df['Iteration Number'] == iter_num) & (merged_df['test_train'] == 'base_train'))]
            num_actives_in_base_train = base_train_row.iloc[0]['supp_Active']
#            found_percent_list.append(1.0-float(num_actives_remaining/(num_actives_in_base_train+num_actives_remaining)))
            found_percent_list.append(num_actives_in_base_train/(num_actives_in_base_train+num_actives_remaining))
            
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
            if classifier == 'random':
                percent_pred_active_list.append(0.0)
            else:
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
def plot_avg_percent_found(merged_df,title,start_size,iter_size):
    '''Takes concated dfs for three seperate runs and plots the average percent recovered over each
    for each classifier across all the AIDs'''
    melted = merged_df.melt(id_vars = ['AID','Classifier','Iteration Number','Embedding','test_train'],var_name = 'Metric',value_name='Score')
    active_find_plot = melted[(melted['Metric'].isin(['Percent Active Found']))&
                        (melted['test_train']=='test')]
    active_find_plot.Score = active_find_plot.Score.astype(float)
    tick_labels = [start_size+iter_size*x for x in list(np.unique(melted['Iteration Number']))]
    plt.figure()
    plt.ioff()
    hue_ord_list = ['SVM','RF','LGBM','DNN','GCNN_pytorch','random']
    #check length of unique classifer names, used as index
    hue_ord_list = hue_ord_list[:len(np.unique(melted.Classifier))]    
    test=sns.lineplot(x="Iteration Number", y="Score",hue='Classifier',
                      data = active_find_plot,
                      ci=68,err_style='band', hue_order = hue_ord_list)
    plt.suptitle(title)
    plt.ylabel('Actives Recovered')
    plt.xlabel('% Library Scanned')
    plt.xticks(np.arange(len(tick_labels)),labels=tick_labels)
    plt.show()
def plot_avg_percent_found_vs_scanned(merged_df,title,start_size,iter_size):
    '''Takes concated dfs for three seperate runs and plots the average percent recovered over each
    for each classifier across all the AIDs'''
    melted = merged_df.melt(id_vars = ['AID','Classifier','Iteration Number','Embedding','test_train','Percent_lib_scanned'],var_name = 'Metric',value_name='Score')
    active_find_plot = melted[(melted['Metric'].isin(['Percent Active Found']))&
                        (melted['test_train']=='test')]
    active_find_plot.Score = active_find_plot.Score.astype(float)
    tick_labels = [start_size+iter_size*x for x in list(np.unique(melted['Iteration Number']))]
    plt.figure()
    plt.ioff()
    hue_ord_list = ['SVM','RF','LGBM','DNN','GCNN_pytorch','random']
    #check length of unique classifer names, used as index
    hue_ord_list = hue_ord_list[:len(np.unique(melted.Classifier))]    
#    test=sns.lineplot(x="Percent_lib_scanned", y="Score",hue='Classifier',size='AID',
#                      data = active_find_plot,
#                      ci=68,err_style='band', hue_order = hue_ord_list)
    test=sns.relplot(x="Percent_lib_scanned", y="Score",hue='Classifier',units='AID',
                      kind='line',  data = active_find_plot,
                      estimator = None,lw=1, col ='Classifier',col_wrap =2,hue_order = hue_ord_list)
#    col ='Classifier',col_wrap =2,
    plt.suptitle(title)
    plt.ylabel('Actives Recovered')
    plt.xlabel('% Library Scanned')
#    plt.xticks(np.arange(len(tick_labels)),labels=tick_labels)
    plt.show()
def get_checkpointsdf(merged_df,start_size,iter_size):
    '''Calculates the percent of actives found at 30% and 50% of the library'''
    #first correct the found percents to delta percent found
    correct_found_list = []
    for _,row in merged_df.iterrows():
        if row['Iteration Number'] != 0:
            AID = row['AID']
            classifier = row['Classifier']
            start_row=merged_df[((merged_df['AID']==AID) & (merged_df['Classifier'] == classifier) 
                                         & (merged_df['Iteration Number'] == 0) & (merged_df['test_train'] == 'test'))]
            start_per = start_row['Percent Active Found'].iloc[0]
            curr_per = row['Percent Active Found']
            correct_found_list.append(curr_per - start_per)
        else:
            correct_found_list.append(0)
    merged_df['Delta Active Found'] = correct_found_list
                
    melted = merged_df.melt(id_vars = ['AID','Classifier','Iteration Number','Embedding','test_train'],var_name = 'Metric',value_name='Score')
    #find iter where 30% was scanned. If it went 25->35 then take both, we'll average later
    iter_num_30 = (30-start_size)/iter_size
    if not iter_num_30.is_integer():
        iter_num_30_list = [int(iter_num_30-0.5),int(iter_num_30+0.5)]     
    else:
        iter_num_30_list = [int(iter_num_30)]
    #now slice out rows with the actives found at the appropriate iteration
    actives_found_30 = melted[(melted['Metric'].isin(['Delta Active Found']))&
                    (melted['test_train']=='test') &(melted['Iteration Number'].isin(iter_num_30_list))]      
    actives_found_30['Lib%'] = 30
    iter_num_50_list = [x+(20/iter_size) for x in iter_num_30_list]
    actives_found_50 = melted[(melted['Metric'].isin(['Delta Active Found']))&
                    (melted['test_train']=='test') &(melted['Iteration Number'].isin(iter_num_50_list))]      
    actives_found_50['Lib%'] = 50
    merged_actives = pd.concat([actives_found_30,actives_found_50])
    merged_actives.Score = merged_actives.Score.astype(float)
    return merged_actives
def set_sns_pal(pal_type):
    '''Either sets to paired or to the darked half of paired'''
    if pal_type =='paired':
        sns.set_palette(sns.color_palette("Paired"))
    else:
        sns.set_palette(['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928'])
def plot_auc_recover_rel(merged_df):
    melted = merged_df.melt(id_vars = ['AID','Classifier','Iteration Number','Embedding','test_train','auc'],var_name = 'Metric',value_name='Score')
    active_find_plot = melted[(melted['Metric'].isin(['Percent Active Found']))&
                        (melted['test_train']=='test')]
    active_find_plot.Score = active_find_plot.Score.astype(float)
    hue_ord_list = ['SVM','RF','LGBM','DNN','GCNN_pytorch','random']
    #check length of unique classifer names, used as index
    hue_ord_list = hue_ord_list[:len(np.unique(melted.Classifier))]  
    sns.lmplot(x='auc',y='Score',col='Iteration Number',hue='Classifier',data=active_find_plot,hue_order = hue_ord_list,truncate=True)
def get_checkpoint35(merged_df,start_size,iter_size):
    '''Calculates the percent of actives found at 30% and 50% of the library'''
    #first correct the found percents to delta percent found
    #^not anymore
    correct_found_list = []
    for _,row in merged_df.iterrows():
        if row['Iteration Number'] != 0:
            AID = row['AID']
            classifier = row['Classifier']
            start_row=merged_df[((merged_df['AID']==AID) & (merged_df['Classifier'] == classifier) 
                                         & (merged_df['Iteration Number'] == 0) & (merged_df['test_train'] == 'test'))]
            start_per = start_row['Percent Active Found'].iloc[0]
            curr_per = row['Percent Active Found']
            correct_found_list.append(curr_per)
        else:
            correct_found_list.append(0)
    merged_df['Delta Active Found'] = correct_found_list
                
    melted = merged_df.melt(id_vars = ['AID','Classifier','Iteration Number','Embedding','test_train'],var_name = 'Metric',value_name='Score')
    #find iter where 30% was scanned. If it went 25->35 then take both, we'll average later
    iter_num_30 = (35-start_size)/iter_size
    if not iter_num_30.is_integer():
        iter_num_30_list = [int(iter_num_30-0.5),int(iter_num_30+0.5)]     
    else:
        iter_num_30_list = [int(iter_num_30)]
    #now slice out rows with the actives found at the appropriate iteration
    actives_found_30 = melted[(melted['Metric'].isin(['Delta Active Found']))&
                    (melted['test_train']=='test') &(melted['Iteration Number'].isin(iter_num_30_list))]      
    actives_found_30['Lib%'] = 35
    
    merged_actives = actives_found_30
    merged_actives.Score = merged_actives.Score.astype(float)
    return merged_actives
def get_checkpoint35_relative(merged_df,start_size,iter_size):
    '''Calculates the percent of actives found relative to random and RF 
    at 30% and 50% of the library'''
    #first correct the found percents to delta percent found
    #^not anymore
    rf_relative_list = []
    random_relative_list = []
    for _,row in merged_df.iterrows():
        if row['Iteration Number'] != 0:
            AID = row['AID']
            classifier = row['Classifier']
            iter_num = row['Iteration Number']
            rf_row=merged_df[((merged_df['AID']==AID) & (merged_df['Classifier'] == 'RF') 
                                         & (merged_df['Iteration Number'] == iter_num) & (merged_df['test_train'] == 'test'))]
            rand_row=merged_df[((merged_df['AID']==AID) & (merged_df['Classifier'] == 'random') 
                                         & (merged_df['Iteration Number'] == iter_num) & (merged_df['test_train'] == 'test'))]
            rf_per = rf_row['Percent Active Found'].iloc[0]
            random_per = rand_row['Percent Active Found'].iloc[0]
            curr_per = row['Percent Active Found']
            rf_relative_list.append(curr_per-rf_per)
            random_relative_list.append(curr_per-random_per)

        else:
            rf_relative_list.append(0)
            random_relative_list.append(0)

    merged_df['RF_Rel'] = rf_relative_list
    merged_df['Rand_Rel'] = random_relative_list

                
    melted = merged_df.melt(id_vars = ['AID','Classifier','Iteration Number','Embedding','test_train'],var_name = 'Metric',value_name='Score')
    #find iter where 30% was scanned. If it went 25->35 then take both, we'll average later
    iter_num_30 = (35-start_size)/iter_size
    if not iter_num_30.is_integer():
        iter_num_30_list = [int(iter_num_30-0.5),int(iter_num_30+0.5)]     
    else:
        iter_num_30_list = [int(iter_num_30)]
    #now slice out rows with the actives found at the appropriate iteration
    actives_found_30 = melted[(melted['Metric'].isin(['RF_Rel','Rand_Rel']))&
                    (melted['test_train']=='test') &(melted['Iteration Number'].isin(iter_num_30_list))]      
    actives_found_30['Lib%'] = 35
    
    merged_actives = actives_found_30
    merged_actives.Score = merged_actives.Score.astype(float)
    return merged_actives