# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:25:39 2019

@author: gdrei
"""
import tensorflow as tf
tf.enable_eager_execution()
import pickle
import pandas as pd
import lightgbm as lgb
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as prf
   
from keras import backend as K
from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler, LabelEncoder
def single_roc(y_preds,y_true):
        
        from sklearn.metrics import roc_curve, auc,precision_recall_curve
        fpr, tpr, _ = roc_curve(y_true, y_preds)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
        plt.plot(recall, precision, color='blue',
                 lw=lw, label='Precision vs Recall')
        # show the plot
        plt.legend(loc="lower right")
        #plt.show()
def multi_roc(y_preds,y_true,name,n_classes):
        import collections
        nested_dict = lambda: collections.defaultdict(nested_dict)
        from tensorflow.keras.utils import to_categorical
        from sklearn.metrics import roc_curve, auc
        from scipy import interp
        from itertools import cycle
        lw = 2
        name_store = ['Active', 'Inactive', 'Inconclusive']
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true[:, i].ravel(), y_preds[:, i].ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area
        
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of '+ name_store[i]+'(area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Multi-class ROC for '+name+' Split= '+str(count+1))
        plt.title('Multi-class ROC for '+name)
    
        plt.legend(loc="lower right")
        #plt.show()