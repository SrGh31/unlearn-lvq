import numpy as np
import os, sys
import sklvq
import pandas as pd
import itertools

parts=os.getcwd().split('/')
if parts[-1]=='notebooks':
    parts=parts[:-1]
datapath='/'.join(parts)+'/data/'


def find_min(indices: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to find the minimum distance and the index of this distance."""
    # Set the irrelevant distances to infinity.
    dist_temp = np.where(indices, distances, np.inf)
    # Find the indices of the closest prototype (column)
    i_dist_min = dist_temp.argmin(axis=1)
    # Return the shortest distances and the indices of the prototypes.
    return dist_temp[np.arange(i_dist_min.size), i_dist_min], i_dist_min


def compute_distances(model, unlearn_data, unlearn_labs):

    num_samples = len(unlearn_labs)
    num_prototypes = model.prototypes_labels_.size
    prototypes_labels=model.prototypes_labels_
    labels=unlearn_labs
    if model.distance_type=='squared-euclidean':
        dist_func=sklvq.distances.SquaredEuclidean()
    rel_distances=dist_func(unlearn_data, model)
    
    if num_samples == 1:
        # Faster if num_samples == 1
        ii_same = np.atleast_2d(labels == prototypes_labels)
    elif num_samples < num_prototypes:
        # Faster to go over the labels if there are less than the prototypes.
        ii_same = np.array([label == prototypes_labels for label in labels])
    else:
        # List comprehension of the prototypes. This are all slight improvements to computation
        # time, as list comprehension takes quite some time.
        ii_same = np.transpose([labels == prototype_label for prototype_label in prototypes_labels])
    # For each prototype mark the samples that have a different label
    ii_diff = ~ii_same
    # For each sample find the closest prototype with the same label. Returns distance and index
    # of prototype
    dist_same, i_dist_same = find_min(ii_same, rel_distances)
    # For each sample find the closest prototype with a different label. Returns distance and
    # index of prototype
    dist_diff, i_dist_diff = find_min(ii_diff, rel_distances)
    return dist_same, dist_diff, i_dist_same, i_dist_diff

def samples_unlearn_random(trainset:pd.DataFrame,trainlabs:list, n:int=20,save_samples:int=0):
    """

    """
    unlearn_indices=np.random.randint(0, trainset.shape[0], size=n).tolist()
    unlearn_samples=trainset.iloc[unlearn_indices].copy()
    relearn_indices, relearn_samples=relearn_unlearn_samples(trainset, unlearn_indices,save_samples)
 #   enforce_indices, enforce_samples=samples_enforce_random(trainset, unlearn_indices,trainlabs)
    unlearn_set={'unlearn_indices':unlearn_indices, 'unlearn_samples':unlearn_samples, 'unlearn_labs':trainlabs[unlearn_indices],
                 'relearn_indices':relearn_indices, 'relearn_samples':relearn_samples, 'relearn_labs':trainlabs[relearn_indices],
                # 'enforce_indices':enforce_indices, 'enforce_samples':enforce_samples, 'enforce_labs':trainlabs[relearn_indices]
                }
    if save_samples==1:
        unlearn_samples.insert(0,'og_index',unlearn_indices)
        unlearn_samples.to_csv(datapath+'unlearn_random_samples_%d.csv'%n, index=False)
    return unlearn_set


def samples_unlearn_outliers(trainset:pd.DataFrame,model:sklvq.models, thresh:float=0.6,save_samples:int=0):
    """

    """
    dist_func=sklvq.distances.SquaredEuclidean()
    rel_distances=dist_func(trainset, model)
    normed_distances=rel_distances/rel_distances.sum(axis=1)[:,np.newaxis]
    sorted_dist, sorted_ind=np.sort(normed_distances, axis=1), np.argsort(normed_distances, axis=1)
    closest_dists=sorted_dist[:,1] -sorted_dist[:,0]
    unlearn_indices=np.where(closest_dists<=thresh)[0]
    unlearn_samples=trainset.iloc[unlearn_indices].copy()
    relearn_indices, relearn_samples=relearn_unlearn_samples(trainset, unlearn_indices,save_samples)
    unlearn_set={'unlearn_indices':unlearn_indices, 'unlearn_samples':unlearn_samples,
                 'relearn_indices':relearn_indices, 'relearn_samples':relearn_samples}
    if save_samples==1:
        unlearn_samples.insert(0,'og_index',unlearn_indices)
        unlearn_samples.to_csv(datapath+'unlearn_outlier_samples.csv', index=False)
    return unlearn_set


def samples_enforce_random(trainset:pd.DataFrame, unlearn_indices:list, trainlabs:list):
    """
    """
    nlabs=np.unique(trainlabs[unlearn_indices])
    if len(unlearn_indices)==1:
        searchset_indices=np.setdiff1d(list(np.where(trainlabs==trainlabs[unlearn_indices])[0]), unlearn_indices)
        #trainset.loc[trainlabs==trainlabs[unlearn_indices]].copy()
        enforce_indices=searchset_indices[np.random.randint(0, len(searchset_indices), size=1).tolist()]
    elif len(nlabs)==1:
        n=len(unlearn_indices)
        searchset_indices=np.setdiff1d(list(np.where(trainlabs==nlabs)[0]), unlearn_indices)
        within_class_sel=list(np.random.randint(0, len(searchset_indices), size=n))
        enforce_indices=searchset_indices[within_class_sel]
    else:
        #intializing container to store indices to enforce
        enforce_indices=[]
        for nlab in nlabs:
            n=np.sum(trainlabs[unlearn_indices]==nlab)
            searchset_indices=np.setdiff1d(list(np.where(trainlabs==nlab)[0]), unlearn_indices)
            if len(searchset_indices)==1:
                within_class_sel=list(np.random.randint(0, len(searchset_indices), size=1))
            elif len(searchset_indices)<=n:
                n=len(searchset_indices)
                within_class_sel=list(np.random.randint(0, len(searchset_indices), size=n))
            else:
                within_class_sel=list(np.random.randint(0, len(searchset_indices), size=n))
            perlab_indices=searchset_indices[within_class_sel]#.tolist()        
            enforce_indices.append(perlab_indices)
        enforce_indices=list(itertools.chain(*enforce_indices))
    return enforce_indices, trainset.iloc[enforce_indices]
    
    

def relearn_unlearn_samples(trainset:pd.DataFrame, unlearn_indices,save_samples:int=0):
    """
    """
    relearn_indices=np.setdiff1d(list(range(0, trainset.shape[0])), unlearn_indices)
    relearn_samples=trainset.iloc[relearn_indices].copy()
    if save_samples==1:
        relearn_samples.insert(0,'og_index',relearn_indices)
        relearn_samples.to_csv(datapath+'relearn_samples_%d.csv'%(len(unlearn_indices)), index=False)
    return relearn_indices, relearn_samples
    
    
    if save_unlearn==1:
        unlearn_samples.insert(0,'og_index',unlearn_indices)
        unlearn_samples.to_csv(datapath+'unlearn_samples.csv', index=False)
    return unlearn_indices