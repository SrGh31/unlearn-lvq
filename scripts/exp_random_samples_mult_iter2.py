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
from utils import samples_unlearn_random
# || Dataset name: Breast cancer data ||
from experiment_unlearn_utils import dataset_health
#dname='breastcancer'
#'adult' #'surgical' # 'diabetes'
dname_all=['diabetes', 'surgical', 'banking', 'adult', 'criteo']
dname=dname_all[1]
Xtrain, Ytrain, Xtest, Ytest, features=dataset_health(dname)
#zXtrain, zXtest=data_normalization(Xtrain, Xtest)
zXtrain, zXtest=data_norm_log(Xtrain, Xtest)
###################################################################################
# Model params to compare; 
# Training original model
dist_name, activation_type="squared-euclidean", "identity"
solver_type, solver_params="sgd", {"max_runs": 5, "step_size": np.array([0.05]), # "k": 3,
                                  }
nprots_per_class=1
glvq=model = GLVQ(
    distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
    solver_type=solver_type, solver_params=solver_params,random_state=42)

glvq.fit(zXtrain, Ytrain)
training_info={'setsize':zXtrain.shape[0], 'class_weight':Counter(Ytrain)}
########################################################################################
if (dname=='breastcancer'):
    nopts=[1,5,20,30,50,100]
#nopts=[0.01,0.05,0.1,1,2]
elif (dname=='criteo'):
    nopts=np.ceil(np.array([0.0001, 0.01, 0.1, 0.2])*len(Ytrain)).astype(np.int32)
else:
    nopts=np.ceil(np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.2])*len(Ytrain)).astype(np.int32)
########################################################################################
# Unlearning parameters to compare
# * number of random samples to unlearn n=[1,5,20,30,50]
cint=0
retrained_n, unlearned_n={},{}
for n in nopts:
    glvq_copy=GLVQ(
    distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
    solver_type=solver_type, solver_params=solver_params,random_state=42)
    glvq_copy.fit(zXtrain, Ytrain)
    dev00, max_dev_indx0=compare_fidelity_glvq(glvq, glvq_copy)
    print('Before unlearning: Deviation between original model and its copy:', dev00)
    random_learn_set=samples_unlearn_random(Xtrain, Ytrain, n,0) #trainset:pd.DataFrame,trainlabs:list, n:int=20,save_samples:int=0
    glvq_copy.secure_prototypes_=glvq_copy.prototypes_.copy()
    retrained_iter, unlearned_iter={},{}
    for iter in [0,1,2]:
        if iter>0:
            glvq_copy.prototypes_=glvq_copy.secure_prototypes_.copy()
        unlearn_indices,relearn_indices=random_learn_set['unlearn_indices'], random_learn_set['relearn_indices']
        unlearn_samples,relearn_samples=random_learn_set['unlearn_samples'], random_learn_set['relearn_samples']
        unlearn_labs,relearn_labs=random_learn_set['unlearn_labs'], random_learn_set['relearn_labs']
        #zXretrain, zXretest=data_normalization(Xtrain.iloc[relearn_indices], Xtest)
        zXretrain, zXretest=data_norm_log(Xtrain.iloc[relearn_indices], Xtest)
        #Retraining 
        st=time.time()
        ####################################
        glvq_partial1=GLVQ(distance_type=dist_name, activation_type=activation_type, prototype_n_per_class=nprots_per_class,
            solver_type=solver_type, solver_params=solver_params)
        glvq_partial1.fit(zXretrain, relearn_labs)
        retrained_iter[iter]={'model': glvq_partial1, 'retrain_indices': relearn_indices }
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
        unlearned_iter[iter]={'model': updated_model, 'unlearn_indices': unlearn_indices }
        #########################################
        elapsed_untrain=(time.time()-st_un)/60
        print('n=%d, Elapsed time diff=%3f-%3f'%(n, elapsed_retrain,elapsed_untrain))
        #########################################
        dev02, max_dev_indx02=compare_fidelity_glvq(glvq, updated_model)
        print('After unlearning: Deviation between original and unlearned models:', dev02)
        dev12, max_dev_indx12=compare_fidelity_glvq(glvq_partial1,updated_model)
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
        compare_dict={'num_prot': nprots_per_class, 'n':len(unlearn_indices), 'iter': iter, 
        'Mapping':'0:original; 1:retrain; 2:unlearn',
            'et_retrain':elapsed_retrain, 'et_unlearn':elapsed_untrain, 
            'prot_dev_01': dev01,'prot_dev_02': dev02,'prot_dev_12': dev12, 'prot_dev_02by12':cratio_uo_ur,
            'dev_nAcc_01': perf01['dev_npreds'],  'dev_nAcc_02': perf02['dev_npreds'],  'dev_nAcc_12': perf12['dev_npreds'],
            'Acc_M0': perf02['M1_acc'], 'Acc_M1': perf12['M1_acc'], 'Acc_M2': perf02['M2_acc'], 
            'AUC_M0': perf02['M1_auc'], 'AUC_M1': perf12['M1_auc'], 'AUC_M2': perf02['M2_auc']}
        if (n==nopts[0]) & (iter==0): #'dev_auc_M1M2'
            compare_df=pd.DataFrame.from_dict(data=compare_dict, orient='index').T
        else:
            temp= pd.DataFrame.from_dict(data=compare_dict, orient='index').T
            compare_df=pd.concat([compare_df, temp])
        compare_df.to_csv(resultspath+'%s/%s_random_unlearn_retrain_nprot%d%s.csv'%(dname, dname,nprots_per_class, solver_type), index=False, sep='\t')
        
    retrained_n[cint]={'n':n, 'models':retrained_iter}
    unlearned_n[cint]={'n':n, 'models': unlearned_iter}
    cint+=1
        
model_sets={'original': glvq, 'retrained': retrained_n, 'unlearned': unlearned_n}
modelpath='/'.join(parts)+'/models/'
picklefilename='%s/%s/%s_random_samples_nprot%d%s.pkl'%(modelpath, dname, dname, nprots_per_class, solver_type)
with open(picklefilename, 'wb') as file:
    pickle.dump(model_sets, file)
#compare_df.applymap(lambda x: '%.3f' % x)
print(dname, ' Num prots: ', nprots_per_class)

        
        



