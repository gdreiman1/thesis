# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:05:32 2019

@author: gdrei
"""

#make data specific estimators
    if embedding_type == 'MFP':
        classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=500000, 
                                                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
                                                        validation_fraction=0.1, n_iter_no_change=5, class_weight='balanced', warm_start=False, average=False)
    elif embedding_type == 'MolChars':
        from sklearn.preprocessing import StandardScaler
        classifier = sklearn.pipeline.Pipeline(Standard_Scaler(),SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=500000, 
                                                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
                                                        validation_fraction=0.1, n_iter_no_change=5, class_weight='balanced', warm_start=False, average=False))
    #run both cross validations
    sklearn.model_selection.cross_validate(classifier, X, y=None, groups=None, scoring=None, cv=’warn’, n_jobs=None, verbose=0, 
                                           fit_params=None, pre_dispatch=‘2*n_jobs’, return_train_score=False, return_estimator=False, error_score=’raise-deprecating’)