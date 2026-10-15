import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import root_mean_squared_error
import sklvq
import os, sys
import copy
parts=os.getcwd().split('/')
if parts[-1]=='notebooks':
    parts=parts[:-1]
code_path='/'.join(parts)+'/scripts/'
sys.path.append(code_path)
from utils import compute_distances
from collections import Counter


def get_grad_funcs(model:sklvq.models._glvq.GLVQ):
    """

    """
    if model.distance_type in ['squared-euclidean', 'euclidean','adaptive-squared-euclidean']:
        
        if model.distance_type=='squared-euclidean':
            dist_func=sklvq.distances.SquaredEuclidean()
          #  grad_dist_func=sklvq.distances.SquaredEuclidean.gradient()
        elif model.distance_type=='euclidean':
            dist_func=sklvq.distances.Euclidean()
            #grad_dist_func=sklvq.distances.gradient()
        else :#model.distance_type=='adaptive-squared-euclidean':
            dist_func=sklvq.distances.AdaptiveSquaredEuclidean()
            #grad_dist_func=sklvq.distances.AdaptiveSquaredEuclidean.gradient()
        grad_dict={'dist_func':dist_func#, 'grad_dist_func':grad_dist_func,
                   }
        return grad_dict
    else:
        print('Error! not yet implemented')
    
def prot_init_correction(prot_lab:int, training_info:dict, label_dict:dict, data_dict:dict):
    """

    """
    initial_cw=training_info['class_weight'][prot_lab]
    unlearn_labels, enforce_labels=label_dict['unlearn_labels'], label_dict['enforce_labels']
    unlearn_data, enforce_data=data_dict['unlearn_data'], data_dict['enforce_data']
    new_N, old_N=training_info['setsize']-len(unlearn_labels)+len(enforce_labels), training_info['setsize']
    if np.sum(unlearn_labels==prot_lab)>1:
        init_subtract_cumul=unlearn_data.loc[unlearn_labels==prot_lab].sum(skipna=True).to_numpy()
    elif np.sum(unlearn_labels==prot_lab)==1:
        init_subtract_cumul=unlearn_data.loc[unlearn_labels==prot_lab].to_numpy()
    else:
        init_subtract_cumul=np.zeros(np.shape(unlearn_data.shape[1]))
        
    if np.sum(enforce_labels==prot_lab)>1:
        init_add_cumul=enforce_data.loc[enforce_labels==prot_lab].sum(skipna=True).to_numpy()
    elif np.sum(enforce_labels==prot_lab)==1:
        init_add_cumul=enforce_data.loc[enforce_labels==prot_lab].to_numpy()
    else:
        init_add_cumul=np.zeros(np.shape(enforce_data.shape[1]))
    init_cumul_corr=init_add_cumul-init_subtract_cumul
    return init_cumul_corr

def get_grads(model:sklvq.models._glvq.GLVQ, adapt_data:pd.DataFrame, adapt_labels:list, nprot:int):
    """

    """
    #grad_dict=get_grad_funcs(model)
    dist_same, dist_diff, i_dist_same, i_dist_diff=compute_distances(model, adapt_data, adapt_labels)
    prot_lab=model.prototypes_labels_[nprot]
    idx_same, idx_diff=np.where(i_dist_same==nprot)[0], np.where(i_dist_diff==nprot)[0]
    # Grad wrt activation
    if model.activation_type=='swish':
        act_func=sklvq.activations.Swish(model.activation_params['beta'])
        get_grad_act_func=sklvq.activations.Swish.gradient(act_func, adapt_data)
    elif model.activation_type=='sigmoid':
        act_func=sklvq.activations.Sigmoid(model.activation_params['beta'])
        get_grad_act_func=sklvq.activations.Sigmoid.gradient(act_func, adapt_data)
    else:
        act_func=sklvq.activations.Identity()
        get_grad_act_func=sklvq.activations.Identity.gradient(adapt_data)
    #get_activation_grad=grad_act_func(act_func,adapt_data)
    # Grad wrt Dist func
    if model.distance_type=='squared-euclidean':
        dist_func=sklvq.distances.SquaredEuclidean()
        get_grad_same_dist=sklvq.distances.SquaredEuclidean.gradient(dist_func, data=adapt_data.iloc[idx_same], model=model, i_prototype=nprot)
        get_grad_diff_dist=sklvq.distances.SquaredEuclidean.gradient(dist_func, data=adapt_data.iloc[idx_diff], model=model, i_prototype=nprot)
    elif model.distance_type=='euclidean':
        dist_func=sklvq.distances.Euclidean()
        get_grad_same_dist=sklvq.distances.Euclidean.gradient(dist_func, data=adapt_data.iloc[idx_same], model=model, i_prototype=nprot)
        get_grad_diff_dist=sklvq.distances.Euclidean.gradient(dist_func, data=adapt_data.iloc[idx_diff], model=model, i_prototype=nprot)
    else:
        dist_func=sklvq.distances.AdaptiveSquaredEuclidean()
        get_grad_same_dist=sklvq.distances.AdaptiveSquaredEuclidean.gradient(dist_func, data=adapt_data.iloc[idx_same], 
                                                                             model=model, i_prototype=nprot)
        get_grad_diff_dist=sklvq.distances.AdaptiveSquaredEuclidean.gradient(dist_func, data=adapt_data.iloc[idx_diff], 
                                                                             model=model, i_prototype=nprot)
    # Grad wrt Discr
    get_grad_same_discr=sklvq.discriminants.RelativeDistance.gradient(sklvq.discriminants.RelativeDistance(), 
                                                                      dist_same=dist_same, dist_diff=dist_diff, same_label=True)
    get_grad_diff_discr=sklvq.discriminants.RelativeDistance.gradient(sklvq.discriminants.RelativeDistance(), 
                                                                      dist_same=dist_same, dist_diff=dist_diff, same_label=False)    
    # Grad combined
    get_grad_same=(get_grad_same_discr[idx_same]*get_grad_act_func.iloc[idx_same].T).T*get_grad_same_dist
    get_grad_diff=(get_grad_diff_discr[idx_diff]*get_grad_act_func.iloc[idx_diff].T).T*get_grad_diff_dist
    if len(idx_same)>0:
        if len(idx_same)==1:
            grad_same=copy.deepcopy(get_grad_same)
        else:
            grad_same=get_grad_same.mean(skipna=True)
    else:
        grad_same=np.zeros(np.shape(model.prototypes_[0]))
    if len(idx_diff)>0:
        if len(idx_diff)==1:
            grad_diff=copy.deepcopy(get_grad_diff)
        else:
            grad_diff=get_grad_diff.mean(skipna=True)
    else:
        grad_diff=np.zeros(np.shape(model.prototypes_[0]))
    grad_diff=np.reshape(grad_diff, (1,model.prototypes_.shape[1]))
    grad_same=np.reshape(grad_same, (1,model.prototypes_.shape[1]))
    return grad_same, grad_diff


    
    

def update_sample_effect_glvq(model:sklvq.models._glvq.GLVQ, data_dict: dict, label_dict: dict, training_info:dict):
    """
    The sklvq toolbox based trained GLVQ model `model` along with the data to be unlearned, `unlearn_data`, 
    their respective labels,(`unlearn_labels` ), `enforcer' samples (`enforced_data') and their respective labels
    `enforced_labels'), are passed into this function. The dictionary `training_info` contains information about 
    the original training set size and the class strengths of the classes it trained on. 
    
    The function returns the updated model with the effects of the `unlearn_data` removed.
    
    Note: This function has been evaluated for the solver types `lbfgs`, `wgd` and `sgd` (steepest gradient descent`, where learning 
    took place in a single batch. 
    We do not know yet whether unlearning from a model trained via a stochastic gradient descent or mini batch gradient descent would 
    work following the same scheme as ordering of samples play a crucial role in these. 
    """
    ztol=10*(-4)
    if model.solver_type=='wgd':
        k=model.get_params()['solver_params']['k']
        max_runs=model.get_params()['solver_params']['max_runs']
        step_size=model.get_params()['solver_params']['step_size'][0]
    elif model.solver_type=='sgd':
        max_runs=model.get_params()['solver_params']['max_runs']
        step_size=model.get_params()['solver_params']['step_size'][0]#/2
    else:
        max_runs=1
        step_size=model.unlearn_rate_
    max_runs, step_size=1, step_size#*10#*max_runs
    unlearn_data, enforce_data=data_dict['unlearn_data'], data_dict['enforce_data']
    unlearn_labels, enforce_labels=label_dict['unlearn_labels'], label_dict['enforce_labels']
    new_N, old_N= training_info['setsize']-len(unlearn_labels)+len(enforce_labels), training_info['setsize']
    secure_copy=copy.deepcopy(model.prototypes_)    
    for iter in range(0, max_runs): 
        updated_prots=copy.deepcopy(model.prototypes_)
        for nprot in range(0,model.prototypes_labels_.shape[0]):         
            unlearn_get_grad_same, unlearn_get_grad_diff=get_grads(model, unlearn_data, unlearn_labels, nprot)
            enforce_get_grad_same, enforce_get_grad_diff=get_grads(model, enforce_data, enforce_labels, nprot)
            if iter==0: 
                prot_lab=model.prototypes_labels_[nprot]
                prot=np.reshape(updated_prots[nprot], (1,model.prototypes_.shape[1]))
                initial_cw=training_info['class_weight'][prot_lab]
                cw_new=initial_cw-np.sum(unlearn_labels==prot_lab)+np.sum(enforce_labels==prot_lab)
                training_info['class_weight'][prot_lab]=cw_new.copy()
                init_cumul_corr=prot_init_correction(prot_lab, training_info, label_dict, data_dict)
                prot_init_corr=np.reshape((prot*initial_cw-init_cumul_corr)/cw_new, (1,model.prototypes_.shape[1]))
                updated_prots[nprot]=np.reshape(prot_init_corr+step_size*((unlearn_get_grad_same+unlearn_get_grad_diff)),
                                        -(enforce_get_grad_same+enforce_get_grad_diff)), 
                                            (1,model.prototypes_.shape[1]))
                del prot_init_corr,cw_new, initial_cw
            else:    
                updated_prots[nprot]=np.reshape(updated_prots[nprot]+step_size*((unlearn_get_grad_same+unlearn_get_grad_diff)
                                        -(enforce_get_grad_same+enforce_get_grad_diff)), (1,model.prototypes_.shape[1]))
           # updated_prots[nprot]=np.reshape(model.prototypes_[nprot].copy(), (1,model.prototypes_.shape[1]))
            del prot, prot_lab, unlearn_get_grad_diff, unlearn_get_grad_same, enforce_get_grad_diff, enforce_get_grad_same
        model.set_prototypes(updated_prots)
        step_size = step_size / (1 + iter/ max_runs)        
    training_info['setsize']=new_N
    return model, training_info