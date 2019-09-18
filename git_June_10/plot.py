# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:12:27 2019

@author: gdrei
"""
import pickle
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(11.7,8.27)})
pickle_off = open(os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', 'multiple_metrics_df_over_under_dnn_partial.pkl'),'rb')
dnn_part1_table=pickle.load(pickle_off)
pickle_off.close()
pickle_off = open(os.path.join('/home/gabriel/Dropbox/UCL/Thesis/Data/', 'multiple_metrics_df_over_under_dnn_partial_cont.pkl'),'rb')
dnn_part2_table=pickle.load(pickle_off)
pickle_off.close()
pickle_off = open(os.path.join('/home/gabriel/Dropbox/UCL/Thesis/June_10/', 'multiple_metrics_df_over_under.pkl'),'rb')
metrics_table=pickle.load(pickle_off)
pickle_off.close()
pickle_off = open(os.path.join('/home/gabriel/Dropbox/UCL/Thesis/June_10/', 'multiple_metrics_df.pkl'),'rb')
over_under_metrics_table=pickle.load(pickle_off)
pickle_off.close()
# concat the two tables:
merged_metrics = pd.concat([metrics_table,over_under_metrics_table,dnn_part1_table,dnn_part2_table],join = 'inner',ignore_index = True)
#plot mcc for all molcharmfp experiments with spits on classifier type and hue on data size
sns.boxplot(x=metrics_table['Classifier'], y=metrics_table['mcc'], hue=metrics_table['Train Split Size'], 
            data=metrics_table[metrics_table['Embedding']=='MFPMolChars'])

#melt the df so that each metric is recorded on a new row.  
metrics_table_melted = pd.melt(metrics_table,id_vars=['10% Split Number', '80% Split Number', 'AID', 'Classifier',
       'Embedding', 'Split Info', 'Train Split Size'],value_vars = ['prec_Inactive','prec_Active','rec_Inactive','rec_Active','f_1_Inactive','f_1_Active','mcc'],var_name = 'Metric',value_name = 'Score')
facet_object = sns.catplot(x='Classifier',y='Score',hue='Train Split Size',kind='violin',
               col='Embedding',row = 'Metric', data = metrics_table_melted[pd.notna(metrics_table_melted['AID'])])

facet_object = sns.catplot(x='Classifier',y='Score',hue='Train Split Size',kind='violin',
               col='Embedding',row = 'Metric', 
               data = metrics_table_melted.loc[(pd.notna(metrics_table_melted['AID'])) & (metrics_table_melted['Metric'].isin(['mcc', 'rec_Active']))])

# now plotting over/under sampling

def make_over_under_labels(row):
    if 'Over' in row['Split Info']:
        return row['Train Split Size'] + ' OverSample'
    elif 'Under' in row['Split Info']:
         return row['Train Split Size'] + ' UnderSample'   
    else:
         return row['Train Split Size'] + ' BaseRatio'

over_under_metrics_table['Over Under Info'] = over_under_metrics_table.apply(make_over_under_labels, axis = 1)
over_under_melt = pd.melt(over_under_metrics_table,id_vars=['10% Split Number', '80% Split Number', 'AID', 'Classifier',
       'Embedding', 'Split Info', 'Train Split Size','Over Under Info'],value_vars = ['prec_Inactive','prec_Active','rec_Inactive','rec_Active','f_1_Inactive','f_1_Active','mcc'],var_name = 'Metric',value_name = 'Score')
facet_object = sns.catplot(x='Classifier',y='Score',hue='Over Under Info',kind='violin',
               col='Embedding',row = 'Metric', palette = sns.color_palette("Paired"),
               data = over_under_melt.loc[(pd.notna(over_under_melt['AID'])) & (over_under_melt['Metric'].isin(['mcc', 'rec_Active']))])

# now plotting over/under with base ratoi
merged_metrics = merged_metrics.dropna(0)
merged_metrics['Over Under Info'] = merged_metrics.apply(make_over_under_labels, axis = 1)
merge_melt = pd.melt(merged_metrics,id_vars=['10% Split Number', '80% Split Number', 'AID', 'Classifier',
       'Embedding', 'Split Info', 'Train Split Size','Over Under Info'],value_vars = ['prec_Inactive','prec_Active','rec_Inactive','rec_Active','f_1_Inactive','f_1_Active','mcc'],var_name = 'Metric',value_name = 'Score')
facet_object = sns.catplot(x='Classifier',y='Score',hue='Over Under Info',kind='violin',
               col='Embedding',row = 'Metric', palette = sns.color_palette("Paired"),
               data = merge_melt.loc[(pd.notna(merge_melt['AID'])) & (merge_melt['Metric'].isin(['rec_Active', 'prec_Active','f_1_Active','mcc']))],legend_out = False)
plt.savefig('recall_prec_f1_mcc_active_allfour.png', dpi=600, format='png', bbox_inches='tight')

#%%
'''Recoverd calls from history log'''
data = a[pd.notna(a['AID']) and a['Score'] == ('mcc' or 'rec_Active')]
data = a[pd.notna(a['AID']) and (a['Score'] == ('mcc' or 'rec_Active'))]
data = a[pd.notna(a['AID']) & (a['Score'] == ('mcc' or 'rec_Active'))]
data = a.loc[(pd.notna(a['AID'])) & (a['Score'].isin(['mcc', 'rec_Active']))]
data = a.loc[ (a['Score'].isin(['mcc', 'rec_Active']))]
data = a.loc[(pd.notna(a['AID'])) ]
data = a.loc[(pd.notna(a['AID'])) & (a['Score'] == ['mcc', 'rec_Active']))]
data = a.loc[(pd.notna(a['AID'])) & (a['core'].isin(['mcc', 'rec_Active']))]
data = a.loc[(pd.notna(a['AID'])) & (a['Metric'].isin(['mcc', 'rec_Active']))]