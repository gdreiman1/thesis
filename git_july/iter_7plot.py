#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:40:01 2019

@author: gabriel
"""

'''Plotting Iter_7'''
from comet_ml import Experiment
exp = Experiment(api_key="sqMrI9jc8kzJYobRXRuptF5Tj",
                        project_name="iter_plotting", workspace="gdreiman1", disabled = False
                        )
exp.log_code = True
exp.log_other('Hypothesis','''These are my plots from the intial iterations Iter_7 ''')
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_dir = '/home/gabriel/Dropbox/UCL/Thesis/Data'
gcnn_initial = 'second_diverse_GCNN_50epoch_iter_run.pkl'
save_path = os.path.join(data_dir,gcnn_initial)
pickle_off = open(save_path,'rb')
gcnn_initial=pickle.load(pickle_off)
pickle_off.close() 

from iter_plot_help_funcs import find_active_percents,plot_metrics,plot_prec_rec_curve,plot_prec_rec_vs_tresh,plot_avg_percent_found,set_sns_pal
set_sns_pal('unpaired')
for exper in [gcnn_initial]:
    exper = find_active_percents(exper,exp)
    plot_metrics(exper,exp)
plot_avg_percent_found(gcnn_initial,'Mean Active Recovery for \n Initial GCNN Experiment',10,5)
