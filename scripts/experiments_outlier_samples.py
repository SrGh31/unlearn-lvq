import os, sys
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.metrics import roc_auc_score, f1_score
import random
from datetime import date
import time
from collections import Counter
from sklvq import GLVQ
import sklvq
parts=os.getcwd().split('/')#[:-1]
if (parts[-1]=='notebooks') |((parts[-1]=='scripts')):
    parts=parts[:-1]
code_path='/'.join(parts)+'/scripts/'
sys.path.append(code_path)

resultspath='/'.join(parts)+'/results/'
from experiment_unlearn_utils import data_normalization, data_norm_log
from unlearning.unlearn_eval import *
from unlearning.unlearn_lvq import unlearn_sample_effect_glvq
from utils import samples_unlearn_outliers, relearn_unlearn_samples
# || Dataset name: Breast cancer data ||
from experiment_unlearn_utils import dataset_health
#dname='breastcancer'
dname_all=['diabetes', 'surgical', 'banking', 'adult', 'criteo']
dname=dname_all[0]
Xtrain, Ytrain, Xtest, Ytest, features=dataset_health(dname)
#zXtrain, zXtest=data_normalization(Xtrain, Xtest)
zXtrain, zXtest=data_norm_log(Xtrain, Xtest)
###################################################################################
# Model params to compare; 
# * nprots_per_class=[1,2,3]
# Training original model
dist_name, activation_type="squared-euclidean", "identity"
solver_type, solver_params="sgd", {"max_runs": 5, "step_size": np.array([0.05]), # "k": 3,
                                  }
nprots_per_class=2
glvq=model = GLVQ(
    distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
    solver_type=solver_type, solver_params=solver_params,random_state=42)

glvq.fit(zXtrain, Ytrain)
training_info={'setsize':zXtrain.shape[0], 'class_weight':Counter(Ytrain)}
########################################################################################
dist_func=sklvq.distances.SquaredEuclidean()
rel_distances=dist_func(zXtrain, glvq)
normed_distances=rel_distances/rel_distances.sum(axis=1)[:,np.newaxis]
sorted_dist, sorted_ind=np.sort(normed_distances, axis=1), np.argsort(normed_distances, axis=1)
closest_dists=(sorted_dist[:,1] -sorted_dist[:,0])
#thresh_opts=[0.000001,0.00001, 0.0001,0.001,0.01]
if dname=='banking':
    nopts, cint=[0.0001,0.001,0.01, 0.02, 0.05,0.08, 0.1], 0
else:
    nopts, cint=[0.000001,0.00001, 0.0001,0.001,0.01, 0.05, 0.1], 0
########################################################################################
# Unlearning parameters to compare
# * number of random samples to unlearn n=[1,5,20,30,50]
for n in nopts:
    glvq_copy=GLVQ(
    distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
    solver_type=solver_type, solver_params=solver_params,random_state=42)
    glvq_copy.fit(zXtrain, Ytrain)
    dev00, max_dev_indx0=compare_fidelity_glvq(glvq, glvq_copy)
    print('Before unlearning: Deviation between original model and its copy:', dev00)
    #outlier_learn_set=samples_unlearn_outliers(zXtrain,glvq, n,0)
    unlearn_indices=np.where(closest_dists<=n)[0]
    relearn_indices, relearn_samples=relearn_unlearn_samples(Xtrain, unlearn_indices,0)
    if (len(unlearn_indices)==0) | (len(relearn_indices)==0):
        print(nprots_per_class, n, len(unlearn_indices), len(relearn_indices))
        continue
    outlier_learn_set={'unlearn_indices':unlearn_indices, 'unlearn_samples':Xtrain.iloc[unlearn_indices],
                 'relearn_indices':relearn_indices, 'relearn_samples':relearn_samples}
   # outlier_learn_set['relearn_labs']=Ytrain[outlier_learn_set['relearn_indices']]
   # outlier_learn_set['unlearn_labs']=Ytrain[outlier_learn_set['unlearn_indices']]
    #outlier_learn_set=samples_unlearn_outliers(Xtrain,Ytrain, n,0)
    unlearn_indices,relearn_indices=outlier_learn_set['unlearn_indices'], outlier_learn_set['relearn_indices']
    unlearn_samples,relearn_samples=outlier_learn_set['unlearn_samples'], outlier_learn_set['relearn_samples']
    unlearn_labs,relearn_labs=Ytrain[unlearn_indices], Ytrain[relearn_indices]
    #zXretrain, zXretest=data_normalization(Xtrain.iloc[relearn_indices], Xtest)
    zXretrain, zXretest=data_norm_log(Xtrain.iloc[relearn_indices], Xtest)
    #Retraining 
    st=time.time()
    ####################################
    glvq_partial1=GLVQ(distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
        solver_type=solver_type, solver_params=solver_params)
    glvq_partial1.fit(zXretrain, relearn_labs)
    #####################################
    elapsed_retrain=(time.time()-st)/60
    #####################################
    dev01, max_dev_indx01=compare_fidelity_glvq(glvq, glvq_partial1)
    print('After retraining: Deviation between original and retrained models:', dev01)
    ###############################################################################################################
    #Unlearning
    st_un=time.time()
    #########################################
    updated_model=unlearn_sample_effect_glvq(glvq_copy, zXtrain.iloc[unlearn_indices], unlearn_labs, training_info)
    #########################################
    elapsed_untrain=(time.time()-st_un)/60
    print('n=%d, Elapsed time diff=%3f-%3f'%(len(unlearn_indices), elapsed_retrain,elapsed_untrain))
    #########################################
    dev02, max_dev_indx02=compare_fidelity_glvq(glvq, updated_model)
    print('After unlearning: Deviation between original and unlearned models:', dev02)
   # glvq_copy.set_prototypes(updated_prots)
   # dev01, max_dev_indx01=compare_fidelity_glvq(glvq, glvq_copy)
    dev12, max_dev_indx12=compare_fidelity_glvq(glvq_partial1,updated_model)
   # dev02, max_dev_indx02=compare_fidelity_glvq(glvq, glvq_copy)
    print('After unlearning: Deviation between retrained and unlearned models:', dev12)
    ###############################################################################################################
    cratio_uo_ur=dev02/dev12
    print('Dev(prots from original and unlearned models)/Dev(prots from retrained and unlearned models)=%0.03f'%cratio_uo_ur)
    ###############################################################################
    # Compare original (0) vs retrained (1)
    data_dict={'zX_M1':zXtest,'zX_M2':zXretest}
    perf01=compare_perf(glvq, glvq_partial1, data_dict,Ytest)
    # Compare original (0) vs unlearned (2)
    data_dict={'zX_M1':zXtest,'zX_M2':zXretest}
    perf02=compare_perf(glvq, updated_model, data_dict,Ytest)
    # Compare retrained (1) vs vs unlearned (2)
    data_dict={'zX_M1':zXretest,'zX_M2':zXretest}
    perf12=compare_perf(glvq_partial1, updated_model, data_dict,Ytest)
    ##############################################################################
    compare_dict={'num_prot': nprots_per_class, 'n':len(unlearn_indices), 'Mapping':'0:original; 1:retrain; 2:unlearn',
        'et_retrain':elapsed_retrain, 'et_unlearn':elapsed_untrain,
        'prot_dev_01': dev01,'prot_dev_02': dev02,'prot_dev_12': dev12, 
        'prot_dev_02by12':cratio_uo_ur, #'prot_dev_02':1-dev02,  'prot_dev_12':1-dev12,
        'dev_nAcc_01': perf01['dev_npreds'],  'dev_nAcc_02': perf02['dev_npreds'],  'dev_nAcc_12': perf12['dev_npreds'],
        'Acc_M0': perf01['M1_acc'], 'Acc_M1': perf12['M1_acc'], 'Acc_M2': perf12['M2_acc'], 
        'AUC_M0': perf01['M1_auc'], 'AUC_M1': perf12['M1_auc'], 'AUC_M2': perf12['M2_auc'], 
       # 'dev_auc_01': perf01['dev_auc'], 'dev_auc_02': perf02['dev_auc'], 'dev_auc_12': perf12['dev_auc']
                 }
    if (n==nopts[0]) | (cint==0): #'dev_auc_M1M2'
        compare_df=pd.DataFrame.from_dict(data=compare_dict, orient='index').T
        cint+=1
    else:
        temp= pd.DataFrame.from_dict(data=compare_dict, orient='index').T
        compare_df=pd.concat([compare_df, temp])
        cint+=1
        compare_df.to_csv(resultspath+'%s_outlier_unlearn_retrain_nprot%d%s.csv'%(dname,nprots_per_class, solver_type), index=False, sep='\t')
print(dname, ' Num prots: ', nprots_per_class)
#compare_df.applymap(lambda x: '%.3f' % x)

        
        



