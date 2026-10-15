import numpy as np
import pandas as pd
import math
import scipy
import itertools
from experiment_utils import dataset_health
from experiment_utils import data_norm_log
import pickle
import sklvq
from collections import Counter
########################################
# Setting up paths for importing and exporting
########################################
import os, sys
parts=os.getcwd().split('/')
print('path',parts)
if (parts[-1]=='scripts') |(parts[-1]=='notebooks'):
    parts=parts[:-1]
common_path='/'.join(parts)
srcpath=common_path+'/results/'
out_path=srcpath+'metaanalysis/'
figpath=common_path+'/figures/'
modelpath='/'.join(parts)+'/models/'
solver_type='sgd'

from analyse_object_utils import import_model_utils
############################################

def eval_perf_metrics(sliced_df):
    """

    """
    models, subsets=['M1', 'M2'], ['tr', 'te', 'un']
    eval_metrics, nan_counter=[],[]
    sliced_df.replace([np.inf, -np.inf], np.nan,inplace=True)

    for subset in subsets:
        sliced_df['%s_Bacc_M0'%(subset)]=sliced_df['%s_M0_Bacc'%(subset)]*100
        eval_metrics.append('%s_Bacc_M0'%(subset))
        if subset=='tr':    
            N=sliced_df['n_retrain']
        elif subset=='te':
            N=sliced_df['n_test']
        else:
            N=sliced_df['n_unlearned']
        sliced_df['%s_Acc_M0'%(subset)]=sliced_df['%s_M0_nAcc'%(subset)]/N
        eval_metrics.append('%s_Acc_M0'%(subset))
        if subset==subsets[0]:
            sliced_df['speedupX']=sliced_df['et_retrain'].to_numpy()/sliced_df['et_unlearn'].to_numpy()
            eval_metrics.append('speedupX')
            eval_metrics.append('dev_M0M2')
            eval_metrics.append('dev_M0M1')
            eval_metrics.append('dev_M1M2')
        for model in models:
            sliced_df['%s_Acc_%s'%(subset, model)]=sliced_df['%s_%s_nAcc'%(subset, model)]*100/N
            eval_metrics.append('%s_Acc_%s'%(subset, model))
            sliced_df['%s_Bacc_%s'%(subset, model)]=sliced_df['%s_%s_Bacc'%(subset, model)]*100
            eval_metrics.append('%s_Bacc_%s'%(subset, model))
            sliced_df['nan_%s_retain_frac_%s'%(subset, model)], sliced_df['nan_%s_loss_frac_%s'%(subset, model)]=0,0
            sliced_df['nan_%s_gain_frac_%s'%(subset, model)]=0
            sliced_df['%s_retain_frac_%s'%(subset, model)]=sliced_df['%s_retain_corr_%s'%(subset, model)]*100/sliced_df['%s_M0_nAcc'%subset]
            sliced_df['%s_loss_frac_%s'%(subset, model)]=sliced_df['%s_lost_preds_%s'%(subset, model)]*100/sliced_df['%s_M0_nAcc'%subset]
            sliced_df['%s_gain_frac_%s'%(subset, model)]=sliced_df['%s_imp_corr_%s'%(subset, model)]*100/sliced_df['%s_M0_nAcc'%subset]#*100
            sliced_df.loc[sliced_df['%s_M0_nAcc'%(subset)]==0, 'nan_%s_retain_frac_%s'%(subset, model)]=1#np.nan
            sliced_df.loc[sliced_df['%s_M0_nAcc'%(subset)]==0, 'nan_%s_loss_frac_%s'%(subset, model)]=1##np.nan
            sliced_df.loc[sliced_df['%s_M0_nAcc'%(subset)]==0, 'nan_%s_gain_frac_%s'%(subset, model)]=1#np.nan
            sliced_df.loc[sliced_df['%s_M0_nAcc'%(subset)]==0, '%s_retain_frac_%s'%(subset, model)]=0#np.nan
            sliced_df.loc[sliced_df['%s_M0_nAcc'%(subset)]==0, '%s_loss_frac_%s'%(subset, model)]=0 #np.nan
            sliced_df.loc[sliced_df['%s_M0_nAcc'%(subset)]==0, '%s_gain_frac_%s'%(subset, model)]=np.nan
            eval_metrics.append('%s_retain_frac_%s'%(subset, model))
            eval_metrics.append('%s_loss_frac_%s'%(subset, model))
            eval_metrics.append('%s_gain_frac_%s'%(subset, model))
            nan_counter.append('%s_retain_frac_%s'%(subset, model))
            nan_counter.append('%s_loss_frac_%s'%(subset, model))
            nan_counter.append('%s_gain_frac_%s'%(subset, model))
    eval_metrics=np.unique(eval_metrics)
    nan_cols=np.array(['nan_'+ entry for entry in nan_counter])#nan_man_std=np.intersect1d
    nan_res=sliced_df.groupby(['n_unlearned','ppc'])[nan_cols].sum().reset_index()
    mean_res=sliced_df.groupby(['n_unlearned','ppc'])[eval_metrics].mean().reset_index()
    std_res=sliced_df.groupby(['n_unlearned','ppc'])[eval_metrics].std().reset_index()
    col_order=np.array(['n_unlearned', 'ppc', 'speedupX','dev_M0M1','dev_M0M2', 'dev_M1M2', 
               'tr_Acc_M0', 'tr_Acc_M1', 'tr_Acc_M2', 
               'tr_Bacc_M0', 'tr_Bacc_M1', 'tr_Bacc_M2', 'tr_retain_frac_M1', 'tr_retain_frac_M2', 
               'tr_loss_frac_M1', 'tr_loss_frac_M2','tr_gain_frac_M1', 'tr_gain_frac_M2', 
               'te_Acc_M0', 'te_Acc_M1', 'te_Acc_M2', 
               'te_Bacc_M0','te_Bacc_M1','te_Bacc_M2', 'te_retain_frac_M1', 
               'te_retain_frac_M2','te_loss_frac_M1', 'te_loss_frac_M2', 'te_gain_frac_M1', 'te_gain_frac_M2',
               'un_Acc_M0', 'un_Acc_M1', 'un_Acc_M2', 
               'un_Bacc_M0','un_Bacc_M1', 'un_Bacc_M2', 'un_retain_frac_M1', 'un_retain_frac_M2',
               'un_loss_frac_M1', 'un_loss_frac_M2', 'un_gain_frac_M1', 'un_gain_frac_M2'])
    mean_res=mean_res[col_order].copy()
    mean_cols=['mean_'+ entry for entry in eval_metrics]
    std_cols=['std_'+ entry for entry in eval_metrics]
    mean_res.rename(columns=dict(zip(eval_metrics, mean_cols)), inplace=True)
    std_res=std_res[col_order].copy()
    std_res.rename(columns=dict(zip(eval_metrics, std_cols)), inplace=True)
    common_cols=list(np.intersect1d(list(mean_res.columns), list(std_res.columns)))    
    final_col_order=list(itertools.chain(*[ [mean_cols[i], std_cols[i]] for i in range(0, len(mean_cols))]))
    combined=pd.concat([mean_res, std_res[std_cols]], axis=1)
    combined=combined[common_cols+final_col_order]
    for nan_col in nan_counter:
        comb_cols=['_'.join(x.split('_')[1:]) for x in list(combined.columns)]
        idx=np.where(nan_col==np.array(comb_cols))[0]
        insert_loc=idx[0]+2
        combined.insert(insert_loc,'nan_%s'%nan_col , nan_res['nan_%s'%nan_col])
    return combined

def perf_tabulate_plot(combined, purpose):
    """

    """
    print_table={}
    combined['ppc']=combined['ppc'].astype(int)
    for i in np.sort(combined['ppc'].unique()):
        dummy_df=combined.loc[(combined['ppc']==i)].copy()
        ext_tab={}
        for sidx, subset in enumerate(['tr','te','un']):
            measures=['n_unlearned', 'mean_%s_Bacc_M1'%(subset), 'std_%s_Bacc_M1'%(subset),
                  'mean_%s_retain_frac_M1'%(subset), 'std_%s_retain_frac_M1'%(subset),
                  'mean_%s_gain_frac_M1'%(subset), 'std_%s_gain_frac_M1'%(subset),
                  'mean_%s_loss_frac_M1'%(subset), 'std_%s_loss_frac_M1'%(subset),
                  'mean_%s_Bacc_M2'%(subset), 'std_%s_Bacc_M2'%(subset),
                  'mean_%s_retain_frac_M2'%(subset), 'std_%s_retain_frac_M2'%(subset),
                  'mean_%s_gain_frac_M2'%(subset), 'std_%s_gain_frac_M2'%(subset),
                  'mean_%s_loss_frac_M2'%(subset), 'std_%s_loss_frac_M2'%(subset)]
            temp_df=dummy_df[measures].copy()
            Mtype=np.concatenate([np.full((1,temp_df['n_unlearned'].nunique()), 'M1'),
                   np.full((1,temp_df['n_unlearned'].nunique()), 'M2')], axis=1)[0]
            df_reorient=pd.DataFrame(data=np.concatenate([temp_df[measures[1:9]].to_numpy(), temp_df[measures[9:]].to_numpy()]),
                             columns=['_'.join([x.split('_')[0], '_'.join(x.split('_')[2:-1])]) for x in measures[1:9]])
            df_reorient.insert(0, 'n_unlearned', np.concatenate([temp_df['n_unlearned'].to_numpy(),temp_df['n_unlearned'].to_numpy()]))
            df_reorient.insert(0, 'Model', Mtype)
            df_reorient.insert(0, 'subset', np.full((1,2*temp_df['n_unlearned'].nunique()), subset)[0])
            df_reorient.insert(0, 'ppc', np.full((1,2*temp_df['n_unlearned'].nunique()), i)[0])
            ext_tab[subset]=df_reorient.copy()
        print_table[i]=ext_tab.copy()
        del ext_tab
    consolidated=pd.concat([print_table[1]['tr'],print_table[1]['te'],print_table[1]['un'],
                            print_table[2]['tr'],print_table[2]['te'],print_table[2]['un']], axis=0, ignore_index=True)
    consolidated.sort_values(by=['ppc', 'Model', 'n_unlearned','subset'], inplace=True)
    consolidated[consolidated.columns[3:]]=consolidated[consolidated.columns[3:]].round(2)
    consolidated['ppc']=consolidated['ppc'].astype(int)
    if purpose=='plot_fig':
        return consolidated
    else:
        cons_cols=np.unique(['_'.join(x.split('_')[1:]) for x in consolidated.columns[4:]])
        for col in cons_cols:
            consolidated[col]=consolidated['mean_'+col].astype(str)+' ('+consolidated['std_'+col].astype(str)+')'
        print_cols=consolidated[list(consolidated.columns[:4])+list(cons_cols)]
        all_cols=print_cols.columns
        for idx, col in enumerate(all_cols):
            if idx<len(all_cols)-1:
                print_cols.insert(np.where(print_cols.columns==col)[0][0]+1, 'amp-'+str(idx+1),
                          np.full((1,print_cols.shape[0]), '&')[0])
            else:
                print_cols.insert(np.where(print_cols.columns==col)[0][0]+1, ' -',
                          np.full((1,print_cols.shape[0]), '\\ \\')[0])
        print_cols['subset']=print_cols['subset'].map({'tr':'retraining', 'te':'test', 'un':'unlearned'})
        print_cols.sort_values(by=['ppc','n_unlearned','Model', 'subset'], inplace=True)
        return print_cols

def compare_model_params_prot(dname:str, exp_type:str,nprots_per_class:int,model_dict):
    """

    """
    exp_type=model_dict['exp_type']
    Xtrain, Ytrain, Xtest, Ytest, features=dataset_health(dname)
    zXtrain, zXtest=data_norm_log(Xtrain, Xtest)
    forget_retained=import_model_utils(dname,model_dict)  
    M0, M1, M2=forget_retained['M0'],forget_retained['M1'],forget_retained['M2']
    unlearn_indices=forget_retained['unlearned']
    zXunlearn, Yunlearn=zXtrain.iloc[unlearn_indices], Ytrain[unlearn_indices]
    print('Class distribution in the forget set: ', Counter(Yunlearn), unlearn_indices)
    idx=0
    if model_dict['ppc']==1:
        prot_group_dummy=pd.DataFrame(data=np.array([Xtrain.columns,
                        np.exp(M0.prototypes_[idx]).T,   np.exp(M1.prototypes_[idx]).T,   np.exp(M2.prototypes_[idx]).T,
                        np.exp(M0.prototypes_[idx+1]).T, np.exp(M1.prototypes_[idx+1]).T, np.exp(M2.prototypes_[idx+1]).T]).T,
                               columns=['features','M0-0','M1-0', 'M2-0','M0-1','M1-1', 'M2-1'])
    else:
        prot_group_dummy=pd.DataFrame(data=np.array([Xtrain.columns,
                        np.exp(M0.prototypes_[idx]).T,   np.exp(M1.prototypes_[idx]).T,   np.exp(M2.prototypes_[idx]).T,
                        np.exp(M0.prototypes_[idx+1]).T, np.exp(M1.prototypes_[idx+1]).T, np.exp(M2.prototypes_[idx+1]).T,
                        np.exp(M0.prototypes_[idx+2]).T, np.exp(M1.prototypes_[idx+2]).T, np.exp(M2.prototypes_[idx+2]).T,  
                        np.exp(M0.prototypes_[idx+3]).T, np.exp(M1.prototypes_[idx+3]).T, np.exp(M2.prototypes_[idx+3]).T]).T,
                                      columns=['features','M0-00','M1-00', 'M2-00','M0-01','M1-01', 'M2-01',
                                       'M0-10','M1-10', 'M2-10','M0-11','M1-11', 'M2-11'])
    prot_group_dummy['Median-%d'%idx]=Xtrain.loc[Ytrain==idx].median().to_numpy()
    prot_group_dummy['Mean-%d'%idx]=Xtrain.loc[Ytrain==idx].mean().to_numpy()
    idx=idx+1
    prot_group_dummy['Median-%d'%idx]=Xtrain.loc[Ytrain==idx].median().to_numpy()
    prot_group_dummy['Mean-%d'%idx]=Xtrain.loc[Ytrain==idx].mean().to_numpy()
    if nprots_per_class==1:
        temp_dummy=prot_group_dummy[['features','M0-0','M1-0', 'M2-0', 'Median-0', 'Mean-0','M0-1','M1-1', 'M2-1', 'Median-1', 'Mean-1']]
    else:
        temp_dummy=prot_group_dummy[['features','M0-00','M1-00', 'M2-00','M0-01','M1-01', 'M2-01', 'Median-0', 'Mean-0',
                          'M0-10','M1-10', 'M2-10','M0-11','M1-11', 'M2-11', 'Median-1', 'Mean-1']]
    return temp_dummy

