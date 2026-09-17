import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import root_mean_squared_error
import sklvq
import os, sys
parts=os.getcwd().split('/')
if parts[-1]=='notebooks':
    parts=parts[:-1]
code_path='/'.join(parts)+'/scripts/'
sys.path.append(code_path)
from utils import compute_distances
from collections import Counter

def unlearn_sample_effect_glvq(model:sklvq.models._glvq.GLVQ, unlearn_data:pd.DataFrame, 
                               unlearn_labels:list, training_info:dict):
    """
    The sklvq toolbox based trained GLVQ model `model` along with the data to be unlearned, `unlearn_data`,
    and their respective labels (`unlearn_labels` ) are passed into this function.The dictionary `training_info`
    contains information about the original training set size and the class strengths of the classes it trained on. 

    The function returns the updated model with the effects of the `adapt_data` removed.
    
    Note: This function has been evaluated for the solver types `lbfgs`, `wgd` and `sgd` (steepest gradient descent`, where learning 
    took place in a single batch. 
    We do not know yet whether unlearning from a model trained via a stochastic gradient descent or mini batch gradient descent would 
    work following the same scheme as ordering of samples play a crucial role in these. 
    """
    if model.solver_type=='wgd':
        k=model.get_params()['solver_params']['k']
        max_iter=k
        max_runs=model.get_params()['solver_params']['max_runs']
    elif model.solver_type=='sgd':
        print(model.solver_type)
        max_runs=model.get_params()['solver_params']['max_runs']
        max_iter=max_runs
    else:
        max_iter=1
    max_iter, new_N, old_N=1, training_info['setsize']-len(unlearn_labels), training_info['setsize']
    secure_copy=model.prototypes_.copy()
    dist_same, dist_diff, i_dist_same, i_dist_diff=compute_distances(model, unlearn_data, unlearn_labels)
    updated_prots=model.prototypes_.copy()
    for iter in range(0, max_iter):   
        for nprot in range(0,model.prototypes_labels_.shape[0]):
            prot_lab=model.prototypes_labels_[nprot]
           # print(prot_lab, training_info['class_weight'])
            initial_cw=training_info['class_weight'][prot_lab]
            cw_new=initial_cw-np.sum(unlearn_labels==prot_lab)
            prot=np.reshape(model.prototypes_[nprot].copy(), (1,model.prototypes_.shape[1]))
            idx_same, idx_diff=np.where(i_dist_same==nprot)[0], np.where(i_dist_diff==nprot)[0]
            get_grad_same=sklvq.distances.SquaredEuclidean.gradient(sklvq.distances.SquaredEuclidean(), 
                                                                   data=unlearn_data.iloc[idx_same], model=model, i_prototype=nprot)
            get_grad_diff=sklvq.distances.SquaredEuclidean.gradient(sklvq.distances.SquaredEuclidean(), 
                                                                   data=unlearn_data.iloc[idx_diff], model=model, i_prototype=nprot)
            if iter==0:
                if np.sum(unlearn_labels==prot_lab)>1:
                    prot_init_corr=(prot*initial_cw-unlearn_data.loc[unlearn_labels==prot_lab].sum(skipna=True).to_numpy())/cw_new
                    #prot_init_corr=prot-unlearn_data.loc[unlearn_labels==prot_lab].mean(skipna=True).to_numpy()#/cw_new
                elif np.sum(unlearn_labels==prot_lab)==1:
                    prot_init_corr=(prot*initial_cw-unlearn_data.loc[unlearn_labels==prot_lab].to_numpy())/cw_new
                else:
                    prot_init_corr=model.prototypes_[nprot].copy()*0#*initial_cw/cw_new
                prot_init_corr=np.reshape(prot_init_corr, (1,model.prototypes_.shape[1]))
                #updated_prots[nprot]=updated_prots[nprot]-prot_init_corr[0]
                updated_prots[nprot]=prot_init_corr[0]#/cw_new
            if len(idx_same)>0:
                if len(idx_same)==1:
                    unlearn_grad_same=get_grad_same*unlearn_data.iloc[idx_same]
                    unlearn_grad_same_sum=unlearn_grad_same.copy()
                else:
                    unlearn_grad_same=get_grad_same.mean(skipna=True)*unlearn_data.iloc[idx_same].mean(skipna=True)
                    unlearn_grad_dummy_same=get_grad_same*unlearn_data.iloc[idx_same]
                    unlearn_grad_same_sum=np.sum(unlearn_grad_dummy_same, axis=0)
                    del  unlearn_grad_dummy_same
            else:
                unlearn_grad_same_sum=np.zeros(np.shape(prot))
                unlearn_grad_same=np.zeros(np.shape(prot))
            if len(idx_diff)>0:
                if len(idx_diff)==1:
                    unlearn_grad_diff=get_grad_diff*unlearn_data.iloc[idx_diff]
                    unlearn_grad_diff_sum=unlearn_grad_diff.copy()
                else:
                    unlearn_grad_diff=get_grad_diff.mean(skipna=True)*unlearn_data.iloc[idx_diff].mean(skipna=True)
                    unlearn_grad_dummy_diff=get_grad_diff*unlearn_data.iloc[idx_diff]
                    unlearn_grad_diff_sum=np.sum(unlearn_grad_dummy_diff, axis=0)
                    del  unlearn_grad_dummy_diff
            else:
                unlearn_grad_diff_sum=np.zeros(np.shape(prot))
                unlearn_grad_diff=np.zeros(np.shape(prot))
            unlearn_grad_diff=np.reshape(unlearn_grad_diff, (1,model.prototypes_.shape[1]))
            unlearn_grad_same=np.reshape(unlearn_grad_same, (1,model.prototypes_.shape[1]))
            #unlearn_grad_diff_sum=np.reshape(unlearn_grad_diff_sum, (1,model.prototypes_.shape[1]))
            #unlearn_grad_same_sum=np.reshape(unlearn_grad_same_sum, (1,model.prototypes_.shape[1]))
            if model.get_params()['solver_type'] in ["sgd", "wgd"]:
                updated_prots[nprot]=(updated_prots[nprot]-(unlearn_grad_diff-
                                                            unlearn_grad_same)*model.get_params()['solver_params']['step_size'][0])*(old_N/new_N)
            else:
           #    updated_prots[nprot]=(updated_prots[nprot]*initial_cw-(unlearn_grad_diff_sum-
           #                                                                 unlearn_grad_same_sum)*model.unlearn_rate_)/cw_new#
                updated_prots[nprot]=(updated_prots[nprot]-(unlearn_grad_diff-
                                                                            unlearn_grad_same)*model.unlearn_rate_)*(old_N/new_N)
            del unlearn_grad_diff, unlearn_grad_same, unlearn_grad_diff_sum, unlearn_grad_same_sum
            del prot, prot_init_corr, cw_new, initial_cw
    model.set_prototypes(updated_prots)
    #model.normalize_variables(model.prototypes_)
    return model

    
def adapt_sample_effect_glvq(model:sklvq.models._glvq.GLVQ, adapt_data:pd.DataFrame, 
                               adapt_labels:list, adapt_type:str, training_info:dict):
    """
    The sklvq toolbox based trained GLVQ model `model` along with the data to be adapted (unlearned or injected), `adapt_data`,
    and their respective labels (`adapt_labels` ) are passed into the function. The variable `adapt_type` tells whether the adaption
    should be unlearning or injection of the `adapt_data`. The dictionary `training_info` contains information about the original 
    training set size and the class strengths of the classes it trained on. 

    The function returns the updated model with the effects of the `adapt_data` removed from injected, based on `adapt_type`.

    Note: This function has been evaluated for the solver types `lbfgs`, `wgd` and `sgd` (steepest gradient descent`, where learning 
    took place in a single batch. 
    We do not know yet whether unlearning from a model trained via a stochastic gradient descent or mini batch gradient descent would 
    work following the same scheme as ordering of samples play a crucial role in these. 
    """
    max_iter=1
    if adapt_type=='unlearn':
        adapt_factor=1
    else:
        adapt_factor=-1
    new_N, old_N=training_info['setsize']-len(adapt_labels), training_info['setsize']
    secure_copy=model.prototypes_.copy()
    dist_same, dist_diff, i_dist_same, i_dist_diff=compute_distances(model, adapt_data, adapt_labels)
    updated_prots=model.prototypes_.copy()
    for iter in range(0, max_iter):   
        for nprot in range(0,model.prototypes_labels_.shape[0]):
            prot_lab=model.prototypes_labels_[nprot]
            initial_cw=training_info['class_weight'][prot_lab]
            cw_new=initial_cw-np.sum(adapt_labels==prot_lab)
            prot=np.reshape(model.prototypes_[nprot].copy(), (1,model.prototypes_.shape[1]))
            idx_same, idx_diff=np.where(i_dist_same==nprot)[0], np.where(i_dist_diff==nprot)[0]
            get_grad_same=sklvq.distances.SquaredEuclidean.gradient(sklvq.distances.SquaredEuclidean(), 
                                                                   data=adapt_data.iloc[idx_same], model=model, i_prototype=nprot)
            get_grad_diff=sklvq.distances.SquaredEuclidean.gradient(sklvq.distances.SquaredEuclidean(), 
                                                                   data=adapt_data.iloc[idx_diff], model=model, i_prototype=nprot)
        # Approximate difference between initialized prots from original data (class-conditional means) 
        # and conditional means from adapt_data. The difference is then adjusted 
        # (subtracted for unlearning, added for relearning) in the trained model's prototype.
            if iter==0: 
                if np.sum(adapt_labels==prot_lab)>1:
                    prot_init_corr=(prot*initial_cw-adapt_factor*adapt_data.loc[adapt_labels==prot_lab].sum(skipna=True).to_numpy())/cw_new
                elif np.sum(adapt_labels==prot_lab)==1:
                    prot_init_corr=(prot*initial_cw-adapt_factor*adapt_data.loc[adapt_labels==prot_lab].to_numpy())/cw_new
                else:
                    prot_init_corr=model.prototypes_[nprot].copy()*0
                prot_init_corr=np.reshape(prot_init_corr, (1,model.prototypes_.shape[1]))
                updated_prots[nprot]=prot_init_corr[0]
            if len(idx_same)>0:
                if len(idx_same)==1:
                    adapt_grad_same=get_grad_same*adapt_data.iloc[idx_same]
                    adapt_grad_same_sum=adapt_grad_same.copy()
                else:
                    adapt_grad_same=get_grad_same.mean(skipna=True)*adapt_data.iloc[idx_same].mean(skipna=True)
                    adapt_grad_dummy_same=get_grad_same*adapt_data.iloc[idx_same]
                    adapt_grad_same_sum=np.sum(adapt_grad_dummy_same, axis=0)
                    del  adapt_grad_dummy_same
            else:
                adapt_grad_same_sum,adapt_same=np.zeros(np.shape(prot)), np.zeros(np.shape(prot))
            if len(idx_diff)>0:
                if len(idx_diff)==1:
                    adapt_grad_diff=get_grad_diff*adapt_data.iloc[idx_diff]
                    adapt_grad_diff_sum=adapt_grad_diff.copy()
                else:
                    adapt_grad_diff=get_grad_diff.mean(skipna=True)*adapt_data.iloc[idx_diff].mean(skipna=True)
                    adapt_grad_dummy_diff=get_grad_diff*adapt_data.iloc[idx_diff]
                    adapt_grad_diff_sum=np.sum(adapt_grad_dummy_diff, axis=0)
                    del adapt_grad_dummy_diff
            else:
                adapt_grad_diff_sum=np.zeros(np.shape(prot))
                adapt_grad_diff=np.zeros(np.shape(prot))
            adapt_grad_diff=np.reshape(adapt_grad_diff, (1,model.prototypes_.shape[1]))
            adapt_grad_same=np.reshape(adapt_grad_same, (1,model.prototypes_.shape[1]))
            if model.get_params()['solver_type'] in ["sgd", "wgd"]:
                updated_prots[nprot]=(updated_prots[nprot]-adapt_factor*(adapt_grad_diff-
                                                            adapt_grad_same)*model.get_params()['solver_params']['step_size'][0])*(old_N/new_N)
            else:
                updated_prots[nprot]=(updated_prots[nprot]-adapt_factor*(adapt_grad_diff-
                                                                            adapt_grad_same)*model.unlearn_rate_)*(old_N/new_N)
            del adapt_grad_diff, adapt_grad_same, adapt_grad_diff_sum, adapt_grad_same_sum
            del prot, prot_init_corr, cw_new, initial_cw
    model.set_prototypes(updated_prots)
    return model

def unlearn_relearn_sample_glvq(model:sklvq.models._glvq.GLVQ, data_dict: dict, label_dict: dict,
                                adapt_action:list, training_info:dict):
    """
    Samples to be unlearned and / or injected are passed into this function in `data_dict` together with their respective label
    in `label_dict`, following the naming convention: `unlearn` or `inject` preceeding `_data` or `_labels` in the respective 
    dictionaries, based on the action that is required to be performed. The `adapt_action` argument is passed as a list of string
    with action name `unlearn` and/or `inject`.

    The function returns an updated model based on unlearning of specified data or injection of new data or data whose effects are 
    to be enforced, or both. It also returns a container with the changed statistics of the data used in the original training, 
    unlearning and/or injection (`cont_stats_change`), where the 0th key-value pair correspond to the original training data based
    statistic.
    """
    c, cont_stats_change=0, {}
    if len(adapt_action)>1:
        for seq in adapt_action:
            adapt_data, adapt_labels=data_dict['%s_data'%seq], label_dict['%s_labels'%seq]
            adapt_type, old_N=seq, training_info['setsize']
            cont_stats_change[c]={'change_type':None,'stats': training_info}
            if adapt_type=='unlearn':
                old_adapt_fac=-1
            else:
                old_adapt_fac=1
            cw_dist0,cw_adapt0 =training_info['class_weight'], Counter(adapt_labels)
            for lab in cw_adapt0.keys():
                cw_dist0[lab]=cw_dist0[lab]+old_adapt_fac*len(adapt_labels)
            if c==0:
                updated_model=adapt_sample_effect_glvq(model, adapt_data, adapt_labels, adapt_type, training_info)
            else:
                updated_model=adapt_sample_effect_glvq(updated_model, adapt_data, adapt_labels, adapt_type, training_info)
            training_info['setsize']=old_N+old_adapt_fac*len(adapt_labels)
            training_info['class_weight']=cw_dist0.copy()
            c+=1
            cont_stats_change[c]={'change_type':adapt_type,'stats': training_info}
    else:
        adapt_type=adapt_action[0]
        if adapt_type=='unlearn':
            old_adapt_fac=-1
        else:
            old_adapt_fac=1
        adapt_data, adapt_labels=data_dict['%s_data'%adapt_type], label_dict['%s_labels'%adapt_type]
        updated_model=adapt_sample_effect_glvq(model, adapt_data, adapt_labels, adapt_type, training_info)
        cw_dist0, cw_adapt0 =training_info['class_weight'], Counter(adapt_labels)
        for lab in cw_adapt0.keys():
            cw_dist0[lab]=cw_dist0[lab]+old_adapt_fac*len(adapt_labels)
        
        training_info['setsize']=training_info['setsize']+old_adapt_fac*len(adapt_labels)
        training_info['class_weight']=cw_dist0.copy()
        cont_stats_change[c]={'change_type':adapt_type,'stats': training_info}
        return updated_model, cont_stats_change
    
        
                
            
    