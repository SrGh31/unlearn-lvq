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
        zXp=zX.copy()
    else:
        zX=data_dict['zX_M1']
        zXp=data_dict['zX_M2']
        est_model1=model1.predict(zX)
        est_model2=model2.predict(zXp)

    if len(np.unique(Y))>2:
        M1_bal_acc, M1_auc=balanced_accuracy_score(Y, est_model1), roc_auc_score(Y, model1.predict_proba(zX),multi_class='ovr', average='weighted')
        M2_bal_acc, M2_auc=balanced_accuracy_score(Y, est_model2), roc_auc_score(Y, model2.predict_proba(zXp), multi_class='ovr', average='weighted')
    else:
        M1_bal_acc, M1_auc=balanced_accuracy_score(Y, est_model1), roc_auc_score(Y, model1.predict_proba(zX)[:,1], average='weighted')
        M2_bal_acc, M2_auc=balanced_accuracy_score(Y, est_model2), roc_auc_score(Y, model2.predict_proba(zXp)[:,1], average='weighted')
    dev_perform={
    'dev_npreds':accuracy_score(Y, est_model1, normalize=False)-accuracy_score(Y, est_model2, normalize=False),
        'dev_auc':np.round(M1_auc-M2_auc, 4),
        'M1_acc':np.round(M1_bal_acc, 4), 'M2_acc':np.round(M2_bal_acc,4),  'M1_auc':np.round(M1_auc,4), 'M2_auc':np.round(M2_auc,4)}
        
def compare_perf_3(model0, model1, model2, data_dict,Y):
    """
    Compare performances of three models model0, model1 and model2 based on 
    the number of samples on which their predictions varied, their balanced accuracies, AUCs.
    """
    if len(data_dict.keys())==1:
        zX=data_dict['zX_M1']
        est_model0=model0.predict(zX)
        est_model1=model1.predict(zX)
        est_model2=model2.predict(zX)
        zXp=zX.copy()
    else:
        zX=data_dict['zX_M1']
        zXp=data_dict['zX_M2']
        est_model0=model0.predict(zX)
        est_model1=model1.predict(zXp)
        est_model2=model2.predict(zXp)

    if len(np.unique(Y))>2:
        M0_bal_acc, M0_auc=balanced_accuracy_score(Y, est_model0), roc_auc_score(Y, model0.predict_proba(zXp), multi_class='ovr', average='weighted')
        M1_bal_acc, M1_auc=balanced_accuracy_score(Y, est_model1), roc_auc_score(Y, model1.predict_proba(zX),multi_class='ovr', average='weighted')
        M2_bal_acc, M2_auc=balanced_accuracy_score(Y, est_model2), roc_auc_score(Y, model2.predict_proba(zXp), multi_class='ovr', average='weighted')
    else:
        M0_bal_acc, M0_auc=balanced_accuracy_score(Y, est_model0), roc_auc_score(Y, model0.predict_proba(zXp)[:,1], average='weighted')
        M1_bal_acc, M1_auc=balanced_accuracy_score(Y, est_model1), roc_auc_score(Y, model1.predict_proba(zX)[:,1], average='weighted')
        M2_bal_acc, M2_auc=balanced_accuracy_score(Y, est_model2), roc_auc_score(Y, model2.predict_proba(zXp)[:,1], average='weighted')
    corr_preds_M0, incorr_preds_M0=np.where(Y==est_model0)[0],np.where(Y!=est_model0)[0]
    corr_preds_M1, incorr_preds_M1=np.where(Y==est_model1)[0],np.where(Y!=est_model1)[0]
    corr_preds_M2, incorr_preds_M2=np.where(Y==est_model2)[0],np.where(Y!=est_model2)[0]
    retained_preds01=list(np.intersect1d(corr_preds_M1, corr_preds_M0))
    lost_preds01=list(np.setdiff1d(corr_preds_M0,corr_preds_M1))
    improved_preds01=list(np.setdiff1d(corr_preds_M1, corr_preds_M0))
    retained_preds02=list(np.intersect1d(corr_preds_M2, corr_preds_M0))
    lost_preds02=list(np.setdiff1d(corr_preds_M0,corr_preds_M2))
    improved_preds02=list(np.setdiff1d(corr_preds_M2, corr_preds_M0))
    dev_perform={
    'M0_npreds': accuracy_score(Y, est_model0, normalize=False), 'M1_npreds':accuracy_score(Y, est_model1, normalize=False),
    'M2_npreds': accuracy_score(Y, est_model2, normalize=False),#'dev_auc':np.round(M1_auc-M2_auc, 4),
    'M0_Bacc':np.round(M0_bal_acc,4),  'M1_Bacc':np.round(M1_bal_acc, 4), 'M2_Bacc':np.round(M2_bal_acc,4), 
    'M0_auc':np.round(M0_auc,4), 'M1_auc':np.round(M1_auc,4), 'M2_auc':np.round(M2_auc,4), 
    'ret_corr_pred_M1': len(retained_preds01), 'ret_corr_pred_M2': len(retained_preds02),
    'lost_corr_pred_M1': len(lost_preds01), 'lost_corr_pred_M2': len(lost_preds02), 
    'imp_corr_pred_M1': len(improved_preds01), 'imp_corr_pred_M2': len(improved_preds02)}
    return dev_perform
