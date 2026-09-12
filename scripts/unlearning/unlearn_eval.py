import numpy as np
import time
import pandas as pd
import os, sys
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import root_mean_squared_error as rmse

def compare_fidelity_glvq(model1, model2):
    """
    Compares the differences in the prototypes of two GLVQ or GMLVQ models
    and returns the root mean squared error of their differences, and the 
    sorted indices of the features based on the absolute deviations
    """
    deviation=np.round(rmse(model1.prototypes_,model2.prototypes_), 5)
    max_dev_indx=np.argsort(np.abs(model1.prototypes_-model2.prototypes_), axis=1)
    return deviation, max_dev_indx


def compare_perf(model1, model2, data_dict,Y):
    """
    Compare performances of two models model1 and model2 based on 
    the number of samples on which their predictions varied, 
    qtheir balanced accuracies, AUCs.
    """
    if len(data_dict.keys())==1:
        zX=data_dict['zX_M1']
        est_model1=model1.predict(zX)
        est_model2=model2.predict(zX)
    else:
        zX=data_dict['zX_M1']
        zXp=data_dict['zX_M2']
        est_model1=model1.predict(zX)
        est_model2=model2.predict(zXp)

    if len(np.unique(Y))>2:
        M1_bal_acc, M1_auc=balanced_accuracy_score(Y, est_model1), roc_auc_score(Y, model1.predict_proba(zX),multi_class='ovr', average='weighted')
        M2_bal_acc, M2_auc=balanced_accuracy_score(Y, est_model2), roc_auc_score(Y, model2.predict_proba(zXp), multi_class='ovr', average='weighted')
  #      dev_perform={'dev_npreds':np.sum(est_model1!=est_model2), 'dev_prediction':est_model1==est_model2,
  #      'dev_acc':M1_bal_acc-M2_bal_acc,'dev_auc':M1_auc-M2_auc,
  #      'M1_acc':M1_bal_acc, 'M2_acc':M2_bal_acc,  'M1_auc':M1_auc, 'M2_auc':M2_auc, }
    else:
        M1_bal_acc, M1_auc=balanced_accuracy_score(Y, est_model1), roc_auc_score(Y, model1.predict_proba(zX)[:,1], average='weighted')
        M2_bal_acc, M2_auc=balanced_accuracy_score(Y, est_model2), roc_auc_score(Y, model2.predict_proba(zXp)[:,1], average='weighted')
    dev_perform={
        #'dev_npreds':np.sum(est_model1!=est_model2), 
        'dev_npreds':accuracy_score(Y, est_model1, normalize=False)-accuracy_score(Y, est_model2, normalize=False),
     #   'dev_prediction':est_model1==est_model2,
       # 'dev_acc':np.round((accuracy_score(Y, est_model1, normalize=False)-accuracy_score(Y, est_model2, normalize=False))/len(Y),2),#M1_bal_acc-M2_bal_acc,
        'dev_auc':np.round(M1_auc-M2_auc, 4),
        'M1_acc':np.round(M1_bal_acc, 4), 'M2_acc':np.round(M2_bal_acc,4),  'M1_auc':np.round(M1_auc,4), 'M2_auc':np.round(M2_auc,4)}
    return dev_perform
