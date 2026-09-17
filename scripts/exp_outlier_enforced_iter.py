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
modelpath='/'.join(parts)+'/models/'
from experiment_utils import data_normalization, data_norm_log
from unlearning.unlearn_eval import *
from unlearning.unlearn_lvq import unlearn_sample_effect_glvq, unlearn_relearn_sample_glvq
from utils import samples_unlearn_outliers, relearn_unlearn_samples, samples_enforce_random
# || Dataset name: Breast cancer data ||
from experiment_utils import dataset_health
#dname='breastcancer'
dname_all=['diabetes', 'surgical', 'banking', 'adult', 'criteo']
dname=dname_all[1]
Xtrain, Ytrain, Xtest, Ytest, features=dataset_health(dname)
#zXtrain, zXtest=data_normalization(Xtrain, Xtest)
zXtrain, zXtest=data_norm_log(Xtrain, Xtest)
###################################################################################
# Model params to compare; 
# * nprots_per_class=[1,2,3]
# Training original model
dist_name, activation_type="squared-euclidean", "identity"
solver_type, solver_params="lbfgs", {"max_runs": 5, "step_size": np.array([0.05]),  "k": 3,
                                  }
print(dname, ' outlier ', solver_type)
for nprot in [1, 2, 3]:
    nprots_per_class=nprot
    cint=0
    if solver_type in ['sgd', 'wgd']:
        glvq=model = GLVQ(
            distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
            solver_type=solver_type, solver_params=solver_params,random_state=42)
        glvq_copy=GLVQ(
            distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
            solver_type=solver_type, solver_params=solver_params,random_state=42)
    else:
        glvq=model = GLVQ(
            distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
            solver_type=solver_type,random_state=42)
        glvq_copy=GLVQ(
            distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
            solver_type=solver_type,random_state=42)

    glvq.fit(zXtrain, Ytrain)
    glvq_copy.fit(zXtrain, Ytrain)
    glvq.normalize_variables(glvq.prototypes_)
    glvq_copy.normalize_variables(glvq_copy.prototypes_)
    training_info={'setsize':zXtrain.shape[0], 'class_weight':Counter(Ytrain)}
    ########################################################################################
    dist_func=sklvq.distances.SquaredEuclidean()
    rel_distances=dist_func(zXtrain, glvq)
    normed_distances=rel_distances/rel_distances.sum(axis=1)[:,np.newaxis]
    sorted_dist, sorted_ind=np.sort(normed_distances, axis=1), np.argsort(normed_distances, axis=1)
    closest_dists=(sorted_dist[:,1] -sorted_dist[:,0])
    #print(closest_dists[:10], np.max(closest_dists), normed_distances[1,:])
    if dname=='banking':
        nopts=[0.0001,0.001,0.01, 0.02, 0.05,0.08, 0.1]
    elif dname=='criteo':
        nopts=[0.0001, 0.001, 0.01, 0.05,0.1]
    else:
        nopts=[0.000001,0.00001, 0.0001,0.001,0.01, 0.05, 0.1]
    ########################################################################################
    # Unlearning parameters to compare
    # * number of random samples to unlearn n=[1,5,20,30,50]
    retrained_n,unlearned_n={},{}
    glvq_copy.secure_prototypes_=glvq_copy.prototypes_.copy()
    cint=0
    for n in nopts:
        if cint>0:
            glvq_copy.prototypes_=glvq_copy.secure_prototypes_.copy()
        dev00, max_dev_indx0=compare_fidelity_glvq(glvq, glvq_copy)
        print('Before unlearning: Deviation between original model and its copy:', dev00)
        unlearn_indices=np.where(closest_dists<=n)[0]
        relearn_indices, relearn_samples=relearn_unlearn_samples(Xtrain, unlearn_indices,0)
        if (len(unlearn_indices)==0) | (len(relearn_indices)==0):
            print(nprots_per_class, n, len(unlearn_indices), len(relearn_indices))
            continue
        outlier_learn_set={'unlearn_indices':unlearn_indices, 'unlearn_samples':Xtrain.iloc[unlearn_indices],
                     'relearn_indices':relearn_indices, 'relearn_samples':relearn_samples}
        unlearn_indices,relearn_indices=outlier_learn_set['unlearn_indices'], outlier_learn_set['relearn_indices']
        unlearn_samples,relearn_samples=outlier_learn_set['unlearn_samples'], outlier_learn_set['relearn_samples']
        enforce_indices, enforce_samples=samples_enforce_random(Xtrain, unlearn_indices,Ytrain)
        unlearn_labs,relearn_labs=Ytrain[unlearn_indices], Ytrain[relearn_indices]
        zXretrain=zXtrain.iloc[relearn_indices].copy()
        #Retraining 
        st=time.time()
        ####################################
        if solver_type in ['sgd', 'wgd']:
            glvq_partial1=GLVQ(distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
            solver_type=solver_type, solver_params=solver_params)
        else:
            glvq_partial1=GLVQ(distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class)
        glvq_partial1.fit(zXretrain, relearn_labs)
        glvq_partial1.normalize_variables(glvq_partial1.prototypes_)
        #####################################
        elapsed_retrain=(time.time()-st)/60
        #####################################
        dev01, max_dev_indx01=compare_fidelity_glvq(glvq, glvq_partial1)
        print('After retraining: Deviation between original and retrained models:', dev01)
        ###############################################################################################################
        #Unlearning
        #########################################
        data_dict={'zX_M1':zXretrain}
        if solver_type=='lbfgs':
            data_dict={'zX_M1':zXretrain}
            grad_step_sizes=np.array([0.001, 0.01, 0.1,0.5, 1])
            acc_diff=np.zeros(len(grad_step_sizes))
            print('Appropriate step size for gradient ascent with %s is being searched using relearning set'%solver_type)
            for idx, grad_step in enumerate(grad_step_sizes):
                glvq_copy.unlearn_rate_=grad_step
                updated_model_attempt=unlearn_relearn_sample_glvq(glvq_copy, zXtrain.iloc[unlearn_indices], 
                            unlearn_labs, zXtrain.iloc[enforce_indices], Ytrain[enforce_indices],training_info)
               # updated_model_attempt.normalize_variables(updated_model_attempt.prototypes_)
                # Compare original (0) vs retrained (1)
                perf123=compare_perf_3(glvq, glvq_partial1, updated_model_attempt, data_dict, relearn_labs)
                # ideal scenario: accuracy of retrained model (M1) should be less than that of unlearned model (M2) 
                acc_diff[idx]=perf123['M0_Bacc']-perf123['M2_Bacc'] 
                glvq_copy.prototypes_=glvq_copy.secure_prototypes_.copy()
            sorted_idx=np.argsort(acc_diff)
            glvq_copy.unlearn_rate_=grad_step_sizes[sorted_idx[0]]
        # Unlearning of sample effects
        st_un=time.time()
        unlearned_model=unlearn_relearn_sample_glvq(glvq_copy, zXtrain.iloc[unlearn_indices], unlearn_labs, 
                                                     zXtrain.iloc[enforce_indices], Ytrain[enforce_indices],training_info)
        elapsed_untrain=(time.time()-st_un)/60
        #########################################
        dev02, max_dev_indx02=compare_fidelity_glvq(glvq, unlearned_model)
        dev12, max_dev_indx12=compare_fidelity_glvq(glvq_partial1, unlearned_model)
        print('After unlearning: Deviation between original and unlearned models:', dev02)
        ###############################################################################################################
        cratio_uo_ur=dev02/dev12
        print('Dev(prots from original and unlearned models)/Dev(prots from retrained and unlearned models)=%0.03f'%cratio_uo_ur)
        ###############################################################################
        # Compare original (0) vs retrained (1) vs unlearned (2)
        data_dict_train={'zX_M1':zXretrain}#zXretrain
        perf_train=compare_perf_3(glvq, glvq_partial1, unlearned_model, data_dict_train, relearn_labs)
        data_dict_test={'zX_M1':zXtest}
        perf_test=compare_perf_3(glvq, glvq_partial1, unlearned_model, data_dict_test, Ytest)
        ##############################################################################
        compare_dict={'num_prot': nprots_per_class, 'n':len(unlearn_indices), #'iter': iter, #'prot_dev_02by12':cratio_uo_ur,
            'Mapping':'0:original; 1:retrain; 2:unlearn','et_retrain':elapsed_retrain, 'et_unlearn':elapsed_untrain, 
            'prot_dev_01': dev01,'prot_dev_02': dev02,'prot_dev_12': dev12,
            'n_retrain': len(relearn_labs), 
            'tr_M0_nAcc': perf_train['M0_npreds'], 'tr_M1_nAcc': perf_train['M1_npreds'], 
            'tr_retain_corr_M1':perf_train['ret_corr_pred_M1'], 'tr_lost_preds_M1':perf_train['lost_corr_pred_M1'],
            'tr_imp_corr_M1':perf_train['imp_corr_pred_M1'], 'tr_M2_nAcc': perf_train['M2_npreds'],  
            'tr_retain_corr_M2':perf_train['ret_corr_pred_M2'], 'tr_lost_preds_M2':perf_train['lost_corr_pred_M2'],
            'tr_imp_corr_M2':perf_train['imp_corr_pred_M2'], 
            'tr_M0_Bacc': perf_train['M0_Bacc'],'tr_M1_Bacc': perf_train['M1_Bacc'],'tr_M2_Bacc': perf_train['M2_Bacc'],
            'tr_M0_AUC': perf_train['M0_auc'],'tr_M1_AUC': perf_train['M1_auc'],'tr_M2_AUC': perf_train['M2_auc'],
            'n_test': len(Ytest), 'te_M0_nAcc': perf_test['M0_npreds'],'te_M1_nAcc': perf_test['M1_npreds'],
            'te_retain_corr_M1':perf_test['ret_corr_pred_M1'], 'te_lost_preds_M1':perf_test['lost_corr_pred_M1'],
            'te_imp_corr_M1':perf_test['imp_corr_pred_M1'], 'te_M2_nAcc': perf_test['M2_npreds'],
            'te_retain_corr_M2':perf_test['ret_corr_pred_M2'], 'te_lost_preds_M2':perf_test['lost_corr_pred_M2'],
            'te_imp_corr_M2':perf_test['imp_corr_pred_M2'],
            'te_M0_Bacc': perf_test['M0_Bacc'], 'te_M1_Bacc': perf_test['M1_Bacc'], 'te_M2_Bacc': perf_test['M2_Bacc'],
            'te_M0_AUC': perf_test['M0_auc'],'te_M1_AUC': perf_test['M1_auc'],'te_M2_AUC': perf_test['M2_auc']}
        retrained_n[cint]={'n':n, 'models':glvq_partial1,'retrain_indices': relearn_indices }
        unlearned_n[cint]={'n':n, 'models': unlearned_model, 'unlearn_indices': unlearn_indices }
        if cint==0: #'dev_auc_M1M2'
            compare_df=pd.DataFrame.from_dict(data=compare_dict, orient='index').T
            cint+=1
        else:
            temp=pd.DataFrame.from_dict(data=compare_dict, orient='index').T
            compare_df=pd.concat([compare_df, temp])
            cint+=1
        if cint>0:
            tab_filename='%s%s/%s_outlier_unlearn_enforce_%s_nprot%d.csv'%(resultspath, dname, dname, solver_type, nprots_per_class)
        compare_df.to_csv(tab_filename, index=False, sep='\t')
    if cint>0:
        model_sets={'original': glvq, 'retrained': retrained_n, 'unlearned': unlearned_n}
        picklefilename='%s/%s/%s_outlier_unlearn_enforce_%s_nprot%d.pkl'%(modelpath, dname, dname, solver_type, nprots_per_class)
        with open(picklefilename, 'wb') as file:
            pickle.dump(model_sets, file)
    print(dname, ' Outlier ', solver_type, ' Num prots: ', nprots_per_class)
    #compare_df.applymap(lambda x: '%.3f' % x)