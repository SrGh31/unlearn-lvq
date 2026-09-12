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

def unlearn_sample_effect_glvq(model:sklvq.models._glvq.GLVQ, unlearn_data:pd.DataFrame, 
                               unlearn_labels:list, training_info:dict):
    """
    """
    #data_unlearn=zXtrain.iloc[unlearn_indices].copy()
    if model.solver_type=='wgd':
        k=model.get_params()['solver_params']['k']
        max_iter=k
        max_runs=model.get_params()['solver_params']['max_runs']
    elif model.solver_type=='sgd':
        print(model.solver_type)
        max_runs=model.get_params()['solver_params']['max_runs']
        max_iter=max_runs
    else:
        max_iter=3
    max_iter=1
    for iter in range(0, max_iter):
        dist_same, dist_diff, i_dist_same, i_dist_diff=compute_distances(model, unlearn_data, unlearn_labels)
        updated_prots=model.prototypes_.copy()
        for nprot in range(0,model.prototypes_labels_.shape[0]):
            prot=np.reshape(model.prototypes_[nprot], (1,model.prototypes_.shape[1]))
            if iter==0:
                initial_cw=training_info['class_weight'][model.prototypes_labels_[nprot]]
                cw_new=initial_cw-np.sum(unlearn_labels==nprot)
                if np.sum(unlearn_labels==nprot)>1:
                    prot_init_corr=prot-(prot*initial_cw-unlearn_data.iloc[unlearn_labels==nprot].sum(skipna=True).to_numpy())/cw_new
                elif np.sum(unlearn_labels==nprot)==1:
                    prot_init_corr=prot-(prot*initial_cw-unlearn_data.iloc[unlearn_labels==nprot].to_numpy())/cw_new
                else:
                    prot_init_corr=prot*0
                prot_init_corr=np.reshape(prot_init_corr, (1,model.prototypes_.shape[1]))
                updated_prots[nprot,:]=updated_prots[nprot,:]-prot_init_corr
                #np.reshape(unlearn_data.iloc[].mean(), (1,model.prototypes_.shape[1]))
            idx_same, idx_diff=np.where(i_dist_same==nprot)[0], np.where(i_dist_diff==nprot)[0]
            get_grad_same=sklvq.distances.SquaredEuclidean.gradient(sklvq.distances.SquaredEuclidean(), 
                                                                   data=unlearn_data.iloc[idx_same], model=model, i_prototype=nprot)
            get_grad_diff=sklvq.distances.SquaredEuclidean.gradient(sklvq.distances.SquaredEuclidean(), 
                                                                   data=unlearn_data.iloc[idx_diff], model=model, i_prototype=nprot)
            if len(idx_same)>0:
                if len(idx_same)==1:
                    unlearn_grad_same=get_grad_same*unlearn_data.iloc[idx_same]
                    prot_init_corr=np.reshape(unlearn_data.iloc[idx_same], (1,model.prototypes_.shape[1]))
                else:
                    unlearn_grad_same=get_grad_same.mean()*unlearn_data.iloc[idx_same].mean(skipna=True)
                    prot_init_corr=np.reshape(unlearn_data.iloc[idx_same].mean(skipna=True), (1,model.prototypes_.shape[1]))
            else:
                unlearn_grad_same=np.zeros(np.shape(prot))
            if len(idx_diff)>0:
                if len(idx_diff)==1:
                    unlearn_grad_diff=get_grad_diff*unlearn_data.iloc[idx_diff]
                   # prot_init_corr=np.reshape(unlearn_data.iloc[idx_diff], (1,model.prototypes_.shape[1]))
                else:
                    unlearn_grad_diff=get_grad_diff.mean()*unlearn_data.iloc[idx_diff].mean(skipna=True)
                 #   prot_init_corr=np.reshape(unlearn_data.iloc[idx_diff].mean(), (1,model.prototypes_.shape[1]))
            else:
                unlearn_grad_diff=np.zeros(np.shape(prot))
            unlearn_grad_diff=np.reshape(unlearn_grad_diff, (1,model.prototypes_.shape[1]))
            unlearn_grad_same=np.reshape(unlearn_grad_same, (1,model.prototypes_.shape[1]))
            updated_prots[nprot,:]=updated_prots[nprot,:]+(unlearn_grad_diff-unlearn_grad_same)*model.get_params()['solver_params']['step_size'][0]
        model.set_prototypes(updated_prots)
        model.normalize_variables(model.prototypes_)
    return model
    
