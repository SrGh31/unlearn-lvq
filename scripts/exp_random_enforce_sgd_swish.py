import os, sys
import numpy as np
import pandas as pd
import re
import copy
import pickle
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score
import random
np.random.seed(42)
from datetime import date
import time
from collections import Counter
from sklvq import GLVQ
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
import sklvq
parts=os.getcwd().split('/')#[:-1]
if (parts[-1]=='notebooks') |((parts[-1]=='scripts')):
    parts=parts[:-1]
common_path=filename='/'.join(parts)
code_path=common_path+'/scripts/'
sys.path.append(code_path)
resultspath=common_path+'/results/'
modelpath=common_path+'/models/'
from experiment_utils import data_norm_log
from unlearning.unlearn_eval import *
from unlearning.unlearn_lvq import unlearn_sample_effect_glvq, unlearn_relearn_sample_glvq
from utils import samples_unlearn_random, relearn_unlearn_samples, samples_enforce_random
# || Dataset name: Breast cancer data ||
from experiment_utils import dataset_health
#################  0 ######### 1 ######### 2 ######## 3 ###### 4 ########## 5
dname_all=['breastcancer', 'surgical', 'banking', 'adult', 'diabetes','criteo']
dname=dname_all[4]
Xtrain, Ytrain, Xtest, Ytest, features=dataset_health(dname)
#zXtrain, zXtest=data_normalization(Xtrain, Xtest)
zXtrain, zXtest=data_norm_log(Xtrain, Xtest)
###########################################
beta_dname={'breastcancer':[5,5], 'surgical': [15,5], 'banking': [18,20], 'adult': [15,10], 'diabetes': [10,10],
'criteo':[10,10]}
#beta_dname2={'breastcancer':5,'surgical': 5 'banking': 20, 'adult': 10, 'diabetes': 10 }
#########################################
import logging
logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('basic_logger')
logfilename='%s/logs/info_enforce_random_%s_swish.log'%(common_path, dname)
print(logfilename)
logging.basicConfig(level=logging.INFO,
    filename=logfilename,# format=fmt,
                    filemode='w', )
logging.info('date={}'.format(date))
########################################################################################
nopts=np.ceil(np.array([0.0001, 0.001, 0.01, 0.1])*len(Ytrain)).astype(np.int32)
flag=1
if flag==0:
    if dname=='adult':
        features=['age', 'num-sex', 'race-summary', 'education-num', 'num-marital',
           'fnlwgt', 'capital-gain', 'hours-per-week', 'Reg_US', 'Work_Private']
        zXtrain,zXtest=zXtrain[features].copy(), zXtest[features].copy()
    elif dname=='diabetes':
        features=['num_gender','num_cat_age', 'num_change', 'num_diabetesMed', 'time_in_hospital',
           'num_lab_procedures', 'num_procedures', 'num_medications','number_outpatient', 
           'number_emergency', 'number_inpatient','number_diagnoses', 'num_acarbose', 'num_acetohexamide', 
           'num_insulin','num_chlorpropamide', 'num_citoglipton', 'num_examide', 'num_glimepiride',
           'num_pioglitazone','num_glipizide', 'num_metformin', 'num_glyburide',  'num_rosiglitazone',
           'num_glimepiride-pioglitazone', 'num_metformin-pioglitazone', 'num_glipizide-metformin', 
           'num_glyburide-metformin', 'num_metformin-rosiglitazone', 'num_miglitol', 'num_nateglinide', 
           'num_repaglinide', 'num_tolazamide', 'num_tolbutamide', 'num_troglitazone',
           'Disch_Home/hospice', 'AfricanAmerican', 'Caucasian']
        zXtrain,zXtest=zXtrain[features].copy(), zXtest[features].copy()
    elif dname=='surgical':
        features=['bmi', 'Age','gender','race', 'asa_status', 'baseline_cancer', 'baseline_charlson',
           'baseline_cvd', 'baseline_dementia', 'baseline_diabetes','baseline_digestive', 'baseline_osteoart', 
           'baseline_psych','baseline_pulmonary', 'ahrq_ccs', 'ccsComplicationRate',#'ccsMort30Rate',
           'complication_rsi', 'mortality_rsi', #'dow', 'hour', 'month','moonphase', 'mort30', 
            ]
        zXtrain,zXtest=zXtrain[features].copy(), zXtest[features].copy()
    elif dname=='breastcancer':
        nopts=[1,5,20,30,50,100]
    elif dname=='criteo':
        nopts=np.ceil(np.array([0.0001, 0.01, 0.1, 0.2])*len(Ytrain)).astype(np.int32)
    else:
        print('No feature preset reqd')
if dname=='breastcancer':
    nopts=[1,5,20,30,50,100]
elif dname=='criteo':
    nopts=np.ceil(np.array([0.0001, 0.01, 0.1])*len(Ytrain)).astype(np.int32)
else:
    print('No feature preset reqd')
###################################################################################
# Model params to compare; 
# Training original model
if dname=='criteo':
    iterations=[0,1,2]
    max_runs, step_size=2, np.array([0.075])
else:
    iterations=[0,1,2,3,4]
    max_runs, step_size=5,np.array([0.05])
dist_name, activation_type="squared-euclidean", "swish",# "identity"
solver_type, solver_params="sgd", {"max_runs": max_runs, "step_size": step_size, # "k": 3,
}
print(dname, ' random ', solver_type)
for nprots in [1,2]:
    nprots_per_class, cint=nprots, 0
    activation_params={"beta": beta_dname[dname][nprots-1]}
    retrained_n, adjusted_n={},{}
    while flag==1:
        if solver_type in ['sgd', 'wgd']:
            glvq=model = GLVQ(
                distance_type=dist_name, activation_type=activation_type, activation_params=activation_params,
                prototype_n_per_class=nprots_per_class,
                solver_type=solver_type, solver_params=solver_params)
            if dname=='criteo':
                glvq_copy=copy.deepcopy(glvq)
            else:
                glvq_copy=GLVQ(
                    distance_type=dist_name, activation_type=activation_type,  activation_params=activation_params,
                    prototype_n_per_class=nprots_per_class,
                    solver_type=solver_type, solver_params=solver_params)
        else:
            glvq= GLVQ(
                distance_type=dist_name, activation_type=activation_type,  activation_params=activation_params,
                prototype_n_per_class=nprots_per_class,
                solver_type=solver_type)
            glvq_copy=GLVQ(
                distance_type=dist_name, activation_type=activation_type,  activation_params=activation_params,
                prototype_n_per_class=nprots_per_class,
                solver_type=solver_type)
    
        glvq.fit(zXtrain, Ytrain)
        acc=balanced_accuracy_score(Ytrain, glvq.predict(zXtrain))
        if acc<0.55:
            flag=1
        else:
            flag=0
        print('Acc=', acc)
    #glvq_copy=copy.copy(glvq)
    glvq_copy.fit(zXtrain, Ytrain)
    training_info={'setsize':zXtrain.shape[0], 'class_weight':Counter(Ytrain), 'normalization':'un-log'}
    ########################################################################################
    # Unlearning parameters to compare
    # * number of random samples to unlearn n=[1,5,20,30,50]
    for n in nopts:
       # logging.info('cint=%d, nprots=%d, n=%d'%(cint, nprots, n))
        glvq_copy=copy.deepcopy(glvq)
        if cint==0:
            glvq_copy.secure_prototypes_=glvq.prototypes_.copy()
        else:
            glvq_copy.prototypes_=glvq.prototypes_.copy()
        dev00, max_dev_indx0=compare_fidelity_glvq(glvq, glvq_copy)
        if dev00>=0.000001:
            logging.info('WARNING: Non-zero deviation before unlearning started: dev=%0.03f'%dev00)
        else:
            logging.info('CLEAR: Zero deviation at baseline')
        retrained_iter, adjusted_iter={},{}
        for iter in iterations:
            random_learn_set=samples_unlearn_random(Xtrain, Ytrain, n,0) 
            #if iter>0:
            glvq_copy=copy.deepcopy(glvq) #.prototypes_.copy()
            unlearn_indices,relearn_indices=random_learn_set['unlearn_indices'], random_learn_set['relearn_indices']
            unlearn_samples,relearn_samples=random_learn_set['unlearn_samples'], random_learn_set['relearn_samples']
            unlearn_labs,relearn_labs=random_learn_set['unlearn_labs'], random_learn_set['relearn_labs']
            unlearn_labs,relearn_labs=random_learn_set['unlearn_labs'], random_learn_set['relearn_labs']
            enforce_indices, enforce_samples=samples_enforce_random(Xtrain, unlearn_indices,Ytrain)
            if len(enforce_indices)==len(unlearn_indices):
                logging.info('CLEAR: Enforced by same number of samples as unlearned')
            else:
                logging.info('WARNING: Enforced by %d samples less than unlearned'%(len(enforce_indices)-len(unlearn_indices)))
            zXretrain=zXtrain.iloc[relearn_indices]
            ########################################################################
            #Retraining 
            ########################################################################
            if solver_type in ['sgd', 'wgd']:
                glvq_partial1=GLVQ(distance_type=dist_name, activation_type=activation_type, activation_params=activation_params,
                                   prototype_n_per_class=nprots_per_class,
                solver_type=solver_type, solver_params=solver_params)
            else:
                glvq_partial1=GLVQ(distance_type=dist_name, activation_type=activation_type, activation_params=activation_params, 
                                   prototype_n_per_class=nprots_per_class,
                solver_type=solver_type)
            st=time.time()
            glvq_partial1.fit(zXretrain, relearn_labs)
            elapsed_retrain=(time.time()-st)/60
            #####################################
            retrained_iter[iter]={'model': glvq_partial1, 'retrain_indices': relearn_indices, 'et_retrain':elapsed_retrain }
            #####################################
            dev01, max_dev_indx01=compare_fidelity_glvq(glvq, glvq_partial1)
            print('After retraining: Deviation between original and retrained models:', dev01)
            #####################################
            dev01, max_dev_indx01=compare_fidelity_glvq(glvq, glvq_partial1)
            perf_before_unlearn=balanced_accuracy_score(unlearn_labs, glvq_copy.predict(zXtrain.iloc[unlearn_indices]))
            print('Unlearned samples perf: before unlearn',perf_before_unlearn)
            #################################################################################################
            #Unlearning
            #########################################
            adapt_action=list(['unlearn','enforce'])
            adj_data_dict={'unlearn_data': zXtrain.iloc[unlearn_indices], 'enforce_data': zXtrain.iloc[enforce_indices]
                          }
            adj_label_dict={'unlearn_labels': unlearn_labs, 'enforce_labels': Ytrain[enforce_indices]
                           }
            if solver_type=='lbfgs':
                data_dict={'zX_M1':zXretrain}
                grad_step_sizes=np.array([0.001, 0.01, 0.1,0.5, 1])
                acc_diff=np.zeros(len(grad_step_sizes))
                print('Appropriate step size for gradient ascent with %s is being searched using relearning set'%solver_type)
                for idx, grad_step in enumerate(grad_step_sizes):
                    glvq_copy.unlearn_rate_=grad_step
                    updated_model_attempt, cont_stats_attempt=unlearn_relearn_sample_glvq(glvq_copy,
                                                adj_data_dict, adj_label_dict,adapt_action, training_info)
                    perf123=compare_perf_3(glvq, glvq_partial1, updated_model_attempt, data_dict, relearn_labs)
                    # ideal scenario:
                    # accuracy of retrained model (M1) should be less than that of unlearned model (M2) 
                    acc_diff[idx]=perf123['M0_Bacc']-perf123['M2_Bacc'] 
                    glvq_copy.prototypes_=glvq.prototypes_.copy()
                sorted_idx=np.argsort(acc_diff)
                print('Appropriate step size for gradient ascent with %s is %.3f'%(solver_type, grad_step_sizes[sorted_idx[0]]))
                glvq_copy.unlearn_rate_=grad_step_sizes[sorted_idx[0]]
            # Unlearning of sample effects
            st_un=time.time()
            adjusted_model,cont_stats_change=unlearn_relearn_sample_glvq(glvq_copy, adj_data_dict, adj_label_dict, 
                                                                         adapt_action, training_info)       
            elapsed_untrain=(time.time()-st_un)/60
            #########################################
            adjusted_iter[iter]={'model': copy.deepcopy(adjusted_model), 'model_prots':adjusted_model.prototypes_.copy(),
                                 'unlearn_indices': unlearn_indices, 'enforced_indices':enforce_indices,
                                'cont_stats_change':cont_stats_change}
            #########################################
            logging.info('n=%d/%d, iter=%d, Elapsed time diff=retrain (%3f) -unlearn (%3f)'%(n, nopts[-1],iter, 
                                                                                             elapsed_retrain, elapsed_untrain))
            #########################################
            dev02, max_dev_indx02=compare_fidelity_glvq(glvq, adjusted_model)
            dev12, max_dev_indx12=compare_fidelity_glvq(glvq_partial1,adjusted_model)
            ###############################################################################
            # Compare original (0) vs retrained (1) vs unlearned (2)
            data_dict_train={'zX_M1':zXretrain}#zXretrain
            perf_train=compare_perf_3(glvq, glvq_partial1, adjusted_model, data_dict_train, relearn_labs)
            data_dict_test={'zX_M1':zXtest}
            perf_test=compare_perf_3(glvq, glvq_partial1, adjusted_model, data_dict_test, Ytest)
            data_dict_unlearn={'zX_M1':zXtrain.iloc[unlearn_indices]}
            perf_unlearn=compare_perf_3(glvq, glvq_partial1, adjusted_model, data_dict_unlearn, Ytrain[unlearn_indices])
            ##############################################################################
            print('Unlearned samples perf: Original vs Retrained vs Unlearned',
                   balanced_accuracy_score(unlearn_labs, glvq.predict(zXtrain.iloc[unlearn_indices])),
                  balanced_accuracy_score(unlearn_labs, glvq_partial1.predict(zXtrain.iloc[unlearn_indices])),
                 balanced_accuracy_score(unlearn_labs, adjusted_model.predict(zXtrain.iloc[unlearn_indices])))
            ##############################################################################
            compare_dict={'ppc': nprots_per_class, 'n_unlearn':len(unlearn_indices), 'iter': iter,
            'et_retrain':elapsed_retrain, 'et_unlearn':elapsed_untrain, 
            'dev_M0M1': dev01,'dev_M0M2': dev02,'dev_M1M2': dev12,
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
            'te_M0_AUC': perf_test['M0_auc'],'te_M1_AUC': perf_test['M1_auc'],'te_M2_AUC': perf_test['M2_auc'],
            'un_M0_nAcc': perf_unlearn['M0_npreds'],'un_M1_nAcc': perf_unlearn['M1_npreds'],
            'un_retain_corr_M1':perf_unlearn['ret_corr_pred_M1'], 'un_lost_preds_M1':perf_unlearn['lost_corr_pred_M1'],
            'un_imp_corr_M1':perf_unlearn['imp_corr_pred_M1'], 'un_M2_nAcc': perf_unlearn['M2_npreds'],
            'un_retain_corr_M2':perf_unlearn['ret_corr_pred_M2'], 'un_lost_preds_M2':perf_unlearn['lost_corr_pred_M2'],
            'un_imp_corr_M2':perf_unlearn['imp_corr_pred_M2'],
            'un_M0_Bacc': perf_unlearn['M0_Bacc'], 'un_M1_Bacc': perf_unlearn['M1_Bacc'], 'un_M2_Bacc': perf_unlearn['M2_Bacc'],
            'un_M0_AUC': perf_unlearn['M0_auc'],'te_M1_AUC': perf_unlearn['M1_auc'],'un_M2_AUC': perf_unlearn['M2_auc']
                         }
            del adjusted_model, unlearn_indices, glvq_partial1, relearn_indices, relearn_labs, zXretrain, enforce_indices, enforce_samples
            if (n==nopts[0]) & (iter==0) & (nprots==1): #'dev_auc_M1M2'
                compare_df=pd.DataFrame.from_dict(data=compare_dict, orient='index').T
            else:
                temp= pd.DataFrame.from_dict(data=compare_dict, orient='index').T
                compare_df=pd.concat([compare_df, temp])
            compare_df.to_csv(resultspath+'%s/%s_random_enforced_unlearn_swish_%s_all.csv'%(dname, dname,solver_type), index=False, sep='\t')
        retrained_n[cint]={'n':n, 'models':retrained_iter}
        adjusted_n[cint]={'n':n, 'models': adjusted_iter}
        cint+=1
            #unlearn_enforce
    model_sets={'original': glvq, 'retrained': retrained_n, 'unlearned':adjusted_n}
    picklefilename='%s%s/%s_random_enforced_swish_%s_nprot%d.pkl'%(modelpath, dname, dname, solver_type, nprots_per_class)
    print('Print picklefile path\n', picklefilename)
    with open(picklefilename, 'wb') as file:
        pickle.dump(model_sets, file)
    del model_sets, retrained_n, adjusted_n, glvq, glvq_copy#, compare_df
    flag=1
    print(dname, ' Num prots: ', nprots_per_class, ' solver_type ', solver_type)