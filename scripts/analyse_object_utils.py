import numpy as np
import pandas as pd
import math
import scipy
import itertools
import os, sys
from collections import Counter
import sklvq
from experiment_utils import dataset_health
from experiment_utils import data_norm_log
import pickle
########################################
# Setting up paths for importing and exporting
########################################
parts=os.getcwd().split('/')
print('path',parts)
if (parts[-1]=='scripts') |(parts[-1]=='notebooks'):
    parts=parts[:-1]
common_path='/'.join(parts)
modelpath='/'.join(parts)+'/models/'
solver_type='sgd'
############################################

def import_model_utils(dname:str,model_dict:dict):
    """

    """
    ppc, exp_type=model_dict['ppc'], model_dict['exp_type']
    if exp_type=='random_enforced':
        picklefilename='%s%s/%s_%s_swish_%s_nprot%d.pkl'%(modelpath, dname, dname, exp_type, solver_type, ppc)
    elif exp_type=='random_stable_enf':
        picklefilename='%s%s/%s_%s_swish_%s_nprot%d.pkl'%(modelpath, dname, dname, exp_type, solver_type, ppc)
    elif exp_type=='outlier':
        picklefilename='%s%s/%s_outlier_swish_%s_nprot%d.pkl'%(modelpath, dname, dname, solver_type, ppc)
    else:
        picklefilename='%s%s/%s_random_swish_%s_nprot%d1.pkl'%(modelpath, dname, dname, solver_type, ppc)
    print('pickle filename is:', picklefilename)
    with open(picklefilename, 'rb') as file:
        model_sets=pickle.load(file)
    ####################################################################
    nopts, iter=model_dict['nopt'], model_dict['iter']
    retrained_all, unlearned_all=model_sets['retrained'], model_sets['unlearned'] 
    n=unlearned_all[nopts]['n']
    M0=model_sets['original']   
    M1, M2=retrained_all[nopts]['models'][iter]['model'], unlearned_all[nopts]['models'][iter]['model']
    unlearn_indices=unlearned_all[nopts]['models'][iter]['unlearn_indices']
    relearn_indices=retrained_all[nopts]['models'][iter]['retrain_indices']
    forget_retained={'unlearned':unlearn_indices, 'M0':M0, 'M1':M1, 'M2':M2,'n_unlearned':n,
                    'retained':relearn_indices}
    if (model_dict['exp_type']=='random_enforced')|(model_dict['exp_type']=='random_stable_enf'):
        enforced_indices=unlearned_all[nopts]['models'][iter]['enforced_indices']
        forget_retained['enforced']=enforced_indices
    return forget_retained


def import_forget_retained_sets(dname:str, model_dict):
    """

    """
    Xtrain, Ytrain, Xtest, Ytest, features=dataset_health(dname)
    zXtrain, zXtest=data_norm_log(Xtrain, Xtest)
    forget_retained=import_model_utils(dname, model_dict)
    M0, M1, M2=forget_retained['M0'],forget_retained['M1'],forget_retained['M2']
    unlearn_indices, relearn_indices=forget_retained['unlearned'], forget_retained['retained']
    zXunlearn, Yunlearn=zXtrain.iloc[unlearn_indices], Ytrain[unlearn_indices]
    zXretained, Yretained=zXtrain.iloc[relearn_indices], Ytrain[relearn_indices]
    retain_forget_data={'retained':zXretained, 'unlearned':zXunlearn}
    retain_forget_label={'retained':Yretained, 'unlearned':Yunlearn}
    if model_dict['exp_type']=='random_enforced':
        enforced_indices=forget_retained['enforced']
        zXenforced, Yenforced=zXtrain.iloc[enforced_all], Ytrain[enforced_indices]
        retain_forget_data['enforced']=zXenforced
        retain_forget_labels['enforced']=Yenforced
    return retain_forget_data, retain_forget_label
  #  print('Class distribution in the forget set: ', Counter(Yunlearn))
        
    
        
    