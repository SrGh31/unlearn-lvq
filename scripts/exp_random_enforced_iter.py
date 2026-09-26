import os, sys
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.metrics import roc_auc_score, f1_score
import random
import copy
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
from experiment_utils import data_norm_log
from unlearning.unlearn_eval import *
from unlearning.unlearn_lvq import unlearn_relearn_sample_glvq
from utils import samples_unlearn_random, samples_enforce_random
# || Dataset name: Breast cancer data ||
from experiment_utils import dataset_health
#dname='breastcancer'
#'adult' #'surgical' # 'diabetes'
dname_all=['breastcancer','diabetes', 'surgical', 'banking', 'adult', 'criteo']
dname=dname_all[0]
Xtrain, Ytrain, Xtest, Ytest, features=dataset_health(dname)
#zXtrain, zXtest=data_normalization(Xtrain, Xtest)
zXtrain, zXtest=data_norm_log(Xtrain, Xtest)
########################################################################################
if (dname=='breastcancer'):
    nopts=[2,5,20,30,50]
#nopts=[0.01,0.05,0.1,1,2]
elif (dname=='criteo'):
    nopts=np.ceil(np.array([0.0001, 0.001, 0.01, 0.1])*len(Ytrain)).astype(np.int32)
else:
    nopts=np.ceil(np.array([0.0001, 0.001, 0.01, 0.1, 0.2])*len(Ytrain)).astype(np.int32)
###################################################################################
# Model params to compare; 
# Training original model
dist_name, activation_type="squared-euclidean", "identity"
solver_type, solver_params="sgd", {"max_runs": 5, "step_size": np.array([0.05]), # "k": 3,
}
print(dname, ' random ', solver_type)
for nprots in [1,2]:
    nprots_per_class=nprots
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
    training_info={'setsize':zXtrain.shape[0], 'class_weight':Counter(Ytrain)}
    ########################################################################################
    # Unlearning and enforcing
    cint=0
    retrained_n, adjusted_n={},{}
    for n in nopts:
        if cint==0:
            glvq_copy.secure_prototypes_=glvq.prototypes_.copy()
        else:
            glvq_copy.prototypes_=glvq_copy.secure_prototypes_.copy()
        dev00, max_dev_indx0=compare_fidelity_glvq(glvq, glvq_copy)
        print('Before unlearning: Deviation between original model and its copy:', dev00)
        retrained_iter, adjusted_iter={},{}
        for iter in [0,1,2,4,5]:
            random_learn_set=samples_unlearn_random(Xtrain, Ytrain, n,0) 
            if iter>0:
                glvq_copy.prototypes_=glvq.prototypes_.copy()
            unlearn_indices,relearn_indices=random_learn_set['unlearn_indices'], random_learn_set['relearn_indices']
            unlearn_samples,relearn_samples=random_learn_set['unlearn_samples'], random_learn_set['relearn_samples']
            unlearn_labs,relearn_labs=random_learn_set['unlearn_labs'], random_learn_set['relearn_labs']
            enforce_indices, enforce_samples=samples_enforce_random(Xtrain, unlearn_indices,Ytrain)
            zXretrain=zXtrain.iloc[relearn_indices]
            #Retraining 
            st=time.time()
            print(Counter(Ytrain))
            ####################################
            if solver_type in ['sgd', 'wgd']:
                glvq_partial1=GLVQ(distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
                solver_type=solver_type, solver_params=solver_params)
            else:
                glvq_partial1=GLVQ(distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
                solver_type=solver_type)
            glvq_partial1.fit(zXretrain, relearn_labs)
            retrained_iter[iter]={'model': glvq_partial1, 'retrain_indices': relearn_indices }
            #####################################
            elapsed_retrain=(time.time()-st)/60
            #####################################
            dev01, max_dev_indx01=compare_fidelity_glvq(glvq, glvq_partial1)
         #   print('After retraining: Deviation between original and retrained models:', dev01)
            perf_before_unlearn=balanced_accuracy_score(unlearn_labs, glvq_copy.predict(zXtrain.iloc[unlearn_indices]))
            print('Unlearned samples perf: before unlearn',perf_before_unlearn)
            #################################################################################################
            #Unlearning
            #########################################
            adapt_action=list(['unlearn'])
            adj_data_dict={'unlearn_data': zXtrain.iloc[unlearn_indices], 
                           'enforce_data': zXtrain.iloc[enforce_indices]
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
        #    print(type( Ytrain[enforce_indices]), len(enforce_indices), len(unlearn_indices))
            adjusted_iter[iter]={'model': adjusted_model, 'model_prots':adjusted_model.prototypes_.copy(),
                                 'unlearn_indices': unlearn_indices, 'enforced_indices':enforce_indices,
                                'cont_stats_change':cont_stats_change}
            adjusted_iter[iter]=copy.copy(adjusted_model)
            #########################################
            elapsed_untrain=(time.time()-st_un)/60
            print('n=%d/%d, Elapsed time diff=retrain (%3f) -unlearn (%3f)'%(n, nopts[-1], elapsed_retrain, elapsed_untrain))
            #########################################
            dev02, max_dev_indx02=compare_fidelity_glvq(glvq, adjusted_model)
          #  print('After unlearning: Deviation between original and unlearned models:', dev02)
            dev12, max_dev_indx12=compare_fidelity_glvq(glvq_partial1,adjusted_model)
          #  print('After unlearning: Deviation between retrained and unlearned models:', dev12)
           #######################################################################################
         #   cratio_uo_ur=dev02/dev12
         #   print('Dev(prots from original and unlearned models)/Dev(prots from retrained and unlearned models)=%0.03f'%cratio_uo_ur)
            perf_after_unlearn=balanced_accuracy_score(unlearn_labs, adjusted_model.predict(zXtrain.iloc[unlearn_indices]))
            perf_after_retrain=balanced_accuracy_score(unlearn_labs, glvq_partial1.predict(zXtrain.iloc[unlearn_indices]))
            print('Unlearned samples perf: Retrained vs Unlearned',perf_after_retrain,perf_after_unlearn)
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
            compare_dict={'num_prot': nprots_per_class, 'n':len(unlearn_indices), 'iter': iter, #'prot_dev_02by12':cratio_uo_ur,
            #'Mapping':'0:original; 1:retrain; 2:unlearn',
            'et_retrain':elapsed_retrain, 'et_unlearn':elapsed_untrain, 
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
            'te_M0_AUC': perf_test['M0_auc'],'te_M1_AUC': perf_test['M1_auc'],'te_M2_AUC': perf_test['M2_auc'],
            'un_M0_nAcc': perf_unlearn['M0_npreds'],'un_M1_nAcc': perf_unlearn['M1_npreds'],
            'un_retain_corr_M1':perf_unlearn['ret_corr_pred_M1'], 'un_lost_preds_M1':perf_unlearn['lost_corr_pred_M1'],
            'un_imp_corr_M1':perf_unlearn['imp_corr_pred_M1'], 'un_M2_nAcc': perf_unlearn['M2_npreds'],
            'un_retain_corr_M2':perf_unlearn['ret_corr_pred_M2'], 'un_lost_preds_M2':perf_unlearn['lost_corr_pred_M2'],
            'un_imp_corr_M2':perf_unlearn['imp_corr_pred_M2'],
            'un_M0_Bacc': perf_unlearn['M0_Bacc'], 'un_M1_Bacc': perf_unlearn['M1_Bacc'], 'un_M2_Bacc': perf_unlearn['M2_Bacc'],
            'un_M0_AUC': perf_unlearn['M0_auc'],'te_M1_AUC': perf_unlearn['M1_auc'],'un_M2_AUC': perf_unlearn['M2_auc']
                         }
            if (n==nopts[0]) & (iter==0): #'dev_auc_M1M2'
                compare_df=pd.DataFrame.from_dict(data=compare_dict, orient='index').T
            else:
                temp= pd.DataFrame.from_dict(data=compare_dict, orient='index').T
                compare_df=pd.concat([compare_df, temp])
            del compare_dict, perf_after_unlearn, perf_train, perf_test, perf_unlearn, data_dict_unlearn, adjusted_model
            tab_filename='%s%s/%s_random_unlearn_enforce_%s_nprot%d.csv'%(resultspath, dname, dname, solver_type, nprots_per_class)
            compare_df.to_csv(tab_filename, index=False, sep='\t')
        retrained_n[cint]={'n':n, 'models':retrained_iter}
        adjusted_n[cint]={'n':n, 'models': adjusted_iter}
        cint+=1
            #unlearn_enforce
    model_sets={'original': glvq, 'retrained': retrained_n, 'unlearned':adjusted_n}
    picklefilename='%s%s/%s_random_unlearn_enforce_%s_nprot%d.pkl'%(modelpath, dname, dname, solver_type, nprots_per_class)
    with open(picklefilename, 'wb') as file:
        pickle.dump(model_sets, file)
    #compare_df.applymap(lambda x: '%.3f' % x)
    print(dname, ' Num prots: ', nprots_per_class, ' solver_type ', solver_type)
