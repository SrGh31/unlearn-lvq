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
    ztol=10*(-4)
    if model.solver_type=='wgd':
        k=model.get_params()['solver_params']['k']
        max_iter=k
        max_runs=model.get_params()['solver_params']['max_runs']
        step_size=model.get_params()['solver_params']['step_size'][0]
    elif model.solver_type=='sgd':
        print(model.solver_type)
        max_runs=model.get_params()['solver_params']['max_runs']
        #max_iter=max_runs
        step_size=model.get_params()['solver_params']['step_size'][0]#/2
       # step_size = step_size / (1 + 1/ max_runs)
    else:
        step_size=model.unlearn_rate_
        max_runs=1
    max_runs=1
    new_N, old_N= training_info['setsize']-len(unlearn_labels), training_info['setsize']
    secure_copy=copy.deepcopy(model.prototypes_)
    for iter in range(0, max_runs):   
        updated_prots=copy.deepcopy(model.prototypes_)#.copy()
        dist_same, dist_diff, i_dist_same, i_dist_diff=compute_distances(model, unlearn_data, unlearn_labels)
        for nprot in range(0,model.prototypes_labels_.shape[0]):                          
            prot_lab=model.prototypes_labels_[nprot]
            idx_same, idx_diff=np.where(i_dist_same==nprot)[0], np.where(i_dist_diff==nprot)[0]
            get_grad_same_dist=sklvq.distances.SquaredEuclidean.gradient(sklvq.distances.SquaredEuclidean(), 
                                                    data=unlearn_data.iloc[idx_same], model=model, i_prototype=nprot)
            get_grad_diff_dist=sklvq.distances.SquaredEuclidean.gradient(sklvq.distances.SquaredEuclidean(), 
                                                    data=unlearn_data.iloc[idx_diff], model=model, i_prototype=nprot)
            get_grad_same_discr=sklvq.discriminants.RelativeDistance.gradient(sklvq.discriminants.RelativeDistance(), 
                                                    dist_same=dist_same, dist_diff=dist_diff, same_label=True)
            get_grad_diff_discr=sklvq.discriminants.RelativeDistance.gradient(sklvq.discriminants.RelativeDistance(), 
                                                    dist_same=dist_same, dist_diff=dist_diff, same_label=False)      
            if model.activation_type=='swish':
                get_activation_grad=sklvq.activations.Swish.gradient(sklvq.activations.Swish(model.activation_params['beta']), unlearn_data)
                get_grad_same=(get_grad_same_discr[idx_same]*get_activation_grad.iloc[idx_same].T).T*get_grad_same_dist
                get_grad_diff=(get_grad_diff_discr[idx_diff]*get_activation_grad.iloc[idx_diff].T).T*get_grad_diff_dist
            elif model.activation_type=='sigmoid':
                get_activation_grad=sklvq.activations.Sigmoid.gradient(sklvq.activations.Sigmoid(model.activation_params['beta']), unlearn_data)
                get_grad_same=(get_grad_same_discr[idx_same]*get_activation_grad.iloc[idx_same].T).T*get_grad_same_dist
                get_grad_diff=(get_grad_diff_discr[idx_diff]*get_activation_grad.iloc[idx_diff].T).T*get_grad_diff_dist
            elif model.activation_type=='identity':
                get_grad_same=(get_grad_same_discr[idx_same]*get_grad_same_dist.T).T
                get_grad_diff=(get_grad_diff_discr[idx_diff]*get_grad_diff_dist.T).T
            else:
                print('Error: Not yet implemented, hence going with identity')
                get_grad_same=(get_grad_same_discr[idx_same]*get_grad_same_dist.T).T
                get_grad_diff=(get_grad_diff_discr[idx_diff]*get_grad_diff_dist.T).T
            if iter==0:
                initial_cw=training_info['class_weight'][prot_lab]
                cw_new=initial_cw-np.sum(unlearn_labels==prot_lab)
                prot=np.reshape(model.prototypes_[nprot].copy(), (1,model.prototypes_.shape[1]))
                if np.sum(unlearn_labels==prot_lab)>1:
                    prot_init_corr=(prot*initial_cw-unlearn_data.loc[unlearn_labels==prot_lab].sum(skipna=True).to_numpy())/cw_new
                elif np.sum(unlearn_labels==prot_lab)==1:
                    prot_init_corr=(prot*initial_cw-unlearn_data.loc[unlearn_labels==prot_lab].to_numpy())/cw_new
                else:
                    prot_init_corr=copy.deepcopy(model.prototypes_[nprot])*0#*initial_cw/cw_new
                prot_init_corr=np.reshape(prot_init_corr, (1,model.prototypes_.shape[1]))
                updated_prots[nprot]=prot_init_corr[0]#/cw_new
            else:
                updated_prots[nprot]=np.reshape(updated_prots[nprot], (1,model.prototypes_.shape[1]))
            if len(idx_same)>0:
                if len(idx_same)==1:
                    if training_info['normalization']=='log':
                        log_corr=unlearn_data.iloc[idx_same].transform(np.exp)#.copy()
                        log_corr[log_corr==0]=ztol
                        unlearn_grad_same=get_grad_same/log_corr
                    else:
                        unlearn_grad_same=copy.deepcopy(get_grad_same)#/unlearn_data.iloc[idx_same]
                else:
                    if training_info['normalization']=='log':
                        log_corr=unlearn_data.iloc[idx_same].transform(np.exp)#.mean(skipna=True)
                        log_corr[log_corr==0]=ztol
                        unlearn_grad_same=get_grad_same/log_corr
                        unlearn_grad_same=unlearn_grad_same.mean(skipna=True)
                    else:
                        unlearn_grad_same=get_grad_same.mean(skipna=True)
                        unlearn_grad_dummy_same=get_grad_same
            else:
                unlearn_grad_same_sum=np.zeros(np.shape(updated_prots[nprot]))
                unlearn_grad_same=np.zeros(np.shape(updated_prots[nprot]))
            if len(idx_diff)>0:
                if len(idx_diff)==1:
                    if training_info['normalization']=='log':
                        log_corr=unlearn_data.iloc[idx_diff].transform(np.exp)#.copy()
                        log_corr[log_corr==0]=ztol
                        unlearn_grad_diff=get_grad_diff/log_corr
                    else:
                        unlearn_grad_diff=copy.deepcopy(get_grad_diff)#/unlearn_data.iloc[idx_diff]
                else:
                    if training_info['normalization']=='log':
                        log_corr=unlearn_data.iloc[idx_diff].transform(np.exp)#.mean(skipna=True)
                        log_corr[log_corr==0]=ztol
                        unlearn_grad_diff=get_grad_diff/log_corr#*unlearn_data.iloc[idx_diff].mean(skipna=True)
                        unlearn_grad_diff=unlearn_grad_diff.mean(skipna=True)
                    else:
                        unlearn_grad_diff=get_grad_diff.mean(skipna=True)#*unlearn_data.iloc[idx_diff].mean(skipna=True)
            else:
                unlearn_grad_diff_sum=np.zeros(np.shape(updated_prots[nprot]))
                unlearn_grad_diff=np.zeros(np.shape(updated_prots[nprot]))
            unlearn_grad_diff=np.reshape(unlearn_grad_diff, (1,model.prototypes_.shape[1]))
            unlearn_grad_same=np.reshape(unlearn_grad_same, (1,model.prototypes_.shape[1]))
            #if model.get_params()['solver_type'] in ["sgd", "wgd"]:
            step_size=1
            updated_prots[nprot]=(updated_prots[nprot]+step_size*(unlearn_grad_diff+unlearn_grad_same))
            #else:
            #    updated_prots[nprot]=(updated_prots[nprot]+(unlearn_grad_diff+unlearn_grad_same)*model.unlearn_rate_)
            del unlearn_grad_diff, unlearn_grad_same, idx_same, idx_diff
            if iter==0:
                del prot, prot_init_corr, cw_new, initial_cw
        model.set_prototypes(updated_prots)
        step_size = step_size / (1 + iter/ max_runs)
        del updated_prots,dist_same, dist_diff, i_dist_same, i_dist_diff
    return model


