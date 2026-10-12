import numpy as np
import pandas as pd
import math
import scipy
import itertools
import os, sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import Counter
import sklvq
import pickle
from sklearn.decomposition import PCA
########################################
# Setting up paths for importing and exporting
########################################
parts=os.getcwd().split('/')
#print('path',parts)
if (parts[-1]=='scripts') |(parts[-1]=='notebooks'):
    parts=parts[:-1]
common_path='/'.join(parts)
srcpath=common_path+'/results/'
out_path=srcpath+'metaanalysis/'
figpath=common_path+'/figures/'
modelpath='/'.join(parts)+'/models/'
solver_type='sgd'

############################################
# import user-defined functions
from experiment_utils import dataset_health
from experiment_utils import data_norm_log
from analyse_object_utils import import_model_utils, import_forget_retained_sets
from evaluations import eval_perf_metrics
##########################################

def plot_model_deviations(exp_type:str,deviation_ylims:dict, fig_dict:dict, row_col_map:dict):
    """

    """
    figsize,marker, linestyle,alpha=fig_dict['figsize'],fig_dict['marker'],fig_dict['linestyle'],fig_dict['alpha']
    fs, lw, mew, mfc, ms=fig_dict['fontsize'],fig_dict['linewidth'], fig_dict['mew'], fig_dict['mfc'],fig_dict['ms']
    lcolor, lcolor2=fig_dict['lcolor'], fig_dict['lcolor2']
    legend_loc, legend_ncols=fig_dict['legend_loc'],fig_dict['legend_ncols']
    legend_propsize, legend_box=fig_dict['legend_propsize'], fig_dict['legend_box']
    if exp_type=='random_enforced':
        dname_all=['breastcancer', 'surgical', 'banking', 'adult', 'diabetes']
        figD, axD= plt.subplots(nrows=2, ncols=int((len(dname_all)+1)/2),figsize=(5,3), dpi=100, layout="constrained")
    else:
        dname_all=['breastcancer', 'surgical', 'banking', 'adult', 'diabetes','criteo']
        figD, axD= plt.subplots(nrows=2, ncols=int(len(dname_all)/2),figsize=(5,3), dpi=100, layout="constrained")
    
    for dname in dname_all:
        metapath='%s%s/%s_random_unlearn_swish_%s_all1.csv'%(srcpath,dname, dname,solver_type)
        res_df=pd.read_csv(metapath, sep='\t')
        res_df.rename(columns={'n_unlearn':'n_unlearned'}, inplace=True)
        combined=eval_perf_metrics(res_df)
        combined=combined.loc[~combined['ppc'].isnull()]
        combined['ppc']=combined['ppc'].astype(int)
        for i in np.sort(combined['ppc'].unique()):
            m, ls,nrow, ncol=marker[i-1], linestyle[i-1],row_col_map[dname][0],row_col_map[dname][1] 
            prot_meta_df=combined.loc[(combined['ppc']==i)]
            h2=axD[nrow, ncol].errorbar(np.log(prot_meta_df['n_unlearned'].to_numpy()), prot_meta_df['mean_dev_M0M1'].to_numpy(),
                    yerr=prot_meta_df['std_dev_M0M1'].to_numpy(), marker=m, mfc='none', mec=lcolor[i-1], linestyle='solid',
                    lw=lw, ms=ms, ecolor=lcolor[i-1],c=lcolor[i-1], alpha=alpha[i-1],  label='nProt=%d'%i)
            h3=axD[nrow, ncol].errorbar(np.log(prot_meta_df['n_unlearned'].to_numpy()), prot_meta_df['mean_dev_M0M2'].to_numpy(),
                    yerr=prot_meta_df['std_dev_M0M2'].to_numpy(), marker=m, mfc='none', mec=lcolor2[i-1], linestyle='-.',
                    lw=lw, ms=ms, ecolor=lcolor2[i-1],c=lcolor2[i-1], alpha=alpha[i-1],  label='nProt=%d'%(i-1))
        axD[nrow, ncol].grid()
        axD[nrow, ncol].tick_params(axis='both', which='major', labelsize=fs)
        axD[nrow, ncol].set_title(dname, fontsize=fs)
        axD[nrow, ncol].set_ylim(deviation_ylims[dname])
        P1_line = mlines.Line2D([], [], color='k', marker=marker[0], mfc='none', mec=lcolor[0], markersize=ms+1, 
                                linestyle=linestyle[0],label='M1:M0, ppc=1')
        P2_line = mlines.Line2D([], [], color='k', marker=marker[1], mfc='none', mec=lcolor[1], markersize=ms+1, 
                                linestyle=linestyle[1],label='M1:M0, ppc=2')
        P3_line = mlines.Line2D([], [], color='k', marker=marker[0], mfc='none', mec=lcolor2[0], markersize=ms+1, 
                                linestyle=linestyle[0],label='M2:M0, ppc=1')
        P4_line = mlines.Line2D([], [], color='k', marker=marker[1], mfc='none', mec=lcolor2[1], markersize=ms+1, 
                                linestyle=linestyle[1],label='M2:M0, ppc=2')
        handleorder2=[P1_line, P2_line, P3_line, P4_line]
        figD.legend(handles=handleorder2, loc=legend_loc, ncols=legend_ncols, prop={"size":legend_propsize}, 
                    bbox_to_anchor=(legend_box[0],legend_box[1]))
        figD.supxlabel('Batch size of samples to be unlearned\n (in log scale)', fontsize=fs)
        figD.supylabel('Deviation b/w parameters of M1 & M2 wrt M0', fontsize=fs)
        if exp_type=='random_enforced':
            filename='%sdeviationM1M2_swish_enforced_%s.pdf'%(figpath, solver_type)
        else:
            filename='%sdeviationM1M2_swish_%s.pdf'%(figpath, solver_type)
        figD.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0.01)


def plot_model_speedup(exp_type:str,speedup_ylims:dict, fig_dict:dict, row_col_map:dict):
    """

    """
    figsize,marker, linestyle,alpha=fig_dict['figsize'],fig_dict['marker'],fig_dict['linestyle'],fig_dict['alpha']
    fs, lw, mew, mfc, ms=fig_dict['fontsize'],fig_dict['linewidth'], fig_dict['mew'], fig_dict['mfc'],fig_dict['ms']
    lcolor=fig_dict['lcolor']
    legend_loc, legend_ncols=fig_dict['legend_loc'],fig_dict['legend_ncols']
    legend_propsize, legend_box=fig_dict['legend_propsize'], fig_dict['legend_box']
    if exp_type=='random_enforced':
        dname_all=['breastcancer', 'surgical', 'banking', 'adult', 'diabetes']
        figS, axS= plt.subplots(nrows=2, ncols=int((len(dname_all)+1)/2),figsize=(5,3), dpi=100, layout="constrained")
    else:
        dname_all=['breastcancer', 'surgical', 'banking', 'adult', 'diabetes','criteo']
        figS, axS= plt.subplots(nrows=2, ncols=int(len(dname_all)/2),figsize=(5,3), dpi=100, layout="constrained")
    for dname in dname_all:
        metapath='%s%s/%s_random_unlearn_swish_%s_all1.csv'%(srcpath,dname, dname,solver_type)
        res_df=pd.read_csv(metapath, sep='\t')
        res_df.rename(columns={'n_unlearn':'n_unlearned'}, inplace=True)
        combined=eval_perf_metrics(res_df)
        combined=combined.loc[~combined['ppc'].isnull()]
        combined['ppc']=combined['ppc'].astype(int)
        for i in np.sort(combined['ppc'].unique()):
            m, ls,nrow, ncol=marker[i-1], linestyle[i-1],row_col_map[dname][0],row_col_map[dname][1] 
            prot_meta_df=combined.loc[(combined['ppc']==i)]
            h=axS[nrow, ncol].errorbar(np.log(prot_meta_df['n_unlearned'].to_numpy()), prot_meta_df['mean_speedupX'].to_numpy(),
                yerr=prot_meta_df['std_speedupX'].to_numpy(), marker=m, mfc='none', mec=lcolor[i-1], linestyle=ls,
                lw=lw, ms=ms, ecolor=lcolor[i-1],c=lcolor[i-1], alpha=alpha[i-1],  label='nProt=%d'%i)
        axS[nrow, ncol].grid()
        axS[nrow, ncol].tick_params(axis='both', which='major', labelsize=fs)
        axS[nrow, ncol].set_title(dname, fontsize=fs)
        axS[nrow, ncol].set_ylim(speedup_ylims[dname])
        P1_line = mlines.Line2D([], [], color='k', marker=marker[0], mfc='none', mec=lcolor[0], markersize=ms+1, 
                                linestyle=linestyle[0],label='ppc=1')
        P2_line = mlines.Line2D([], [], color='k', marker=marker[1], mfc='none', mec=lcolor[1], markersize=ms+1, 
                                linestyle=linestyle[1],label='ppc=2')
        handleorder1=[P1_line, P2_line]
        figS.legend(handles=handleorder1, loc=legend_loc, ncols=legend_ncols, prop={"size":legend_propsize}, 
                    bbox_to_anchor=(legend_box[0],legend_box[1]))
        figS.supxlabel('Batch size of samples to be unlearned\n (in log scale)', fontsize=fs)
        figS.supylabel(r'Speed-up factor ($\frac{\text{Time taken to retrain}}{\text{Time taken to unlearn}}$)', fontsize=fs)
        if exp_type=='random_enforced':
            filename='%sspeedup_swish-enforced_%s.pdf'%(figpath, solver_type)
        else:
            filename='%sspeedup_swish%s.pdf'%(figpath, solver_type)
        figS.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0.01)


def plot_perf_per_df(dname:str, consolidated:pd.DataFrame, fig_dict:dict):
    """
    dname: name of the dataset 
    consolidated: DataFrame with the mean and standard deviation of the performance metrics: balanced accuracy, 
    % of retained predictions, % of lost predictions, % of predictions improved upon.
    fig_dict: dictionary containing figure size, markers, marker size, marker face and edge colors, linestyle,
            linewidth, alpha, and legend details such as the prop size, location, legend anchoring position.
    Returns:
        Plots the performance metrics in consolidated, for retrained model (M1) vs unlearned model (M2).

    """
    consolidated['ppc']=consolidated['ppc'].astype(int)
    panel_col_dict={'tr': 'Re-training set', 'te': 'Test set', 'un': 'Unlearned set'}
    vert_panel_col_dict={'retain frac': '% Retained', 'loss frac': '% Lost', 'gain frac': '% Improved upon',
                    'Bacc':'% Bal. Accuracy'}
    measures=consolidated.columns[4:]
    nrows, ncols, figsize,steps, step_rows=fig_dict['nrows'],fig_dict['ncols'], fig_dict['figsize'],fig_dict['steps'], fig_dict['step_rows']
    fs, lw, mew, mfc, ms=fig_dict['fontsize'],fig_dict['linewidth'], fig_dict['mew'], fig_dict['mfc'],fig_dict['ms']
    marker, linestyle, lcolor, alpha=fig_dict['marker'],fig_dict['linestyle'],fig_dict['lcolor'], fig_dict['alpha']
    legend_loc, legend_ncols=fig_dict['legend_loc'],fig_dict['legend_ncols']
    legend_propsize, legend_box=fig_dict['legend_propsize'], fig_dict['legend_box']
    ylims_all=fig_dict['ylims_all']
    fig0, ax0= plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(figsize[0],figsize[1]), layout="constrained")    
    nprots_all=np.sort(consolidated['ppc'].unique())
    for i in nprots_all:
        for sidx, subset in enumerate(['tr','te','un']):
            for midx, model in enumerate(['M1', 'M2']):
                for nrow in range(0,nrows):
                    c=2*nrow
                    colname_parts=measures[c].split('_')[1:]
                    temp_df=consolidated[['n_unlearned',measures[c], measures[c+1]]].loc[(consolidated['ppc']==i) &
                    (consolidated['subset']==subset) &(consolidated['Model']==model)].copy()
                    ax0[nrow, sidx].errorbar(np.log(temp_df['n_unlearned']), temp_df[measures[c]], yerr=temp_df[measures[c+1]], 
                                             color=lcolor[midx], marker=marker[i-1], ls=linestyle[i-1], mew=mew, lw=lw, 
                                             ms=ms, mec=lcolor[midx],mfc='none',label='%s: %d ppc'%(model,i), alpha=alpha[i-1])
                    if (i==1) & (midx==0):
                        ax0[nrow, sidx].grid()    
                    if nrow==0:
                        ylabel=colname_parts[0]
                        if (i==1) & (midx==0):
                            ax0[nrow,sidx].set_title(panel_col_dict[subset], fontsize=fs)
                    else:
                        ylabel=' '.join(colname_parts)
                    c_ylim=ylims_all[ylabel]
                    if nrow in step_rows:
                        step=steps[0]
                    else:
                        step=steps[1]
                    ax0[nrow,sidx].set_ylim(c_ylim)
                    ax0[nrow, sidx].set_yticks(np.arange(c_ylim[0], c_ylim[1], step))
                    ax0[nrow, sidx].tick_params(axis='both', which='both', labelsize=fs)  
                    ax0[nrow, sidx].yaxis.set_tick_params(labelsize=fs-3, rotation=0)
                    if midx==0:                    
                        ax0[nrow,0].set_ylabel(vert_panel_col_dict[ylabel], fontsize=fs)                 
    handles, labels = ax0[0,0].get_legend_handles_labels()
    patch1,patch2 = mpatches.Patch(color=lcolor[0], label='M1'), mpatches.Patch(color=lcolor[1], label='M2')
    P1_line = mlines.Line2D([], [], color='k', marker=marker[0], mfc='none', mec='k', markersize=ms+1, linestyle=linestyle[0],label='ppc=1')
    if dname=='criteo':
        handleorder1=[patch1, patch2, P1_line]
    else:
        P2_line = mlines.Line2D([], [], color='k', marker=marker[1], mfc='none', mec='k', markersize=ms+1, linestyle=linestyle[1],label='ppc=2')
        handleorder1, handleorder2=[patch1, patch2, P1_line, P2_line], [patch1, P1_line, patch2, P2_line]
    fig0.legend(handles=handleorder1, loc=legend_loc, ncols=legend_ncols, prop={"size":legend_propsize},
                bbox_to_anchor=(legend_box[0], legend_box[1]))
    fig0.supxlabel(r'Batch size of $\mathcal{D}^f$ (in log scale) $\rightarrow$', fontsize=fs+1)
    fig0.suptitle(dname.capitalize(), fontsize=fs+1)
    if 'suffix' in fig_dict:
        filename='%sperf_swish_wrt_M0_%s_%s_%s.pdf'%(figpath, dname, solver_type, fig_dict['suffix'])
    else:
        filename='%sperf_swish_wrt_M0_%s_%s.pdf'%(figpath, dname, solver_type)
    fig0.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0.01)


def plot_prototypes_compare(dname:str, model_dict:dict, fig_dict:dict):
    """
    dname: name of the dataset 
    model_dict: dictionary containing the iteration number (iter), number of unlearned samples (nopt),
                the number of prototype per class (ppc), and the experiment type (exp_type)
    fig_dict: dictionary containing figure size, markers, marker size, marker face and edge colors, linestyle,
            linewidth, alpha, and legend details such as the prop size, location, legend anchoring position.
    Returns:
        Plots the prototypes from the original model (M0), retrained model (M1), and unlearned model (M2)
    """
    ####################################################################
    # importing the data to plot unlearned/ enforced/ relearned samples
    Xtrain, Ytrain, Xtest, Ytest, features=dataset_health(dname)
    feature_names=Xtrain.columns
#    zXtrain, zXtest=data_norm_log(Xtrain, Xtest)
    ###################################################################
    # importing the spedific models 
    ####################################################################  
    ppc, exp_type=model_dict['ppc'], model_dict['exp_type']
    forget_retained=import_model_utils(dname,model_dict)  
    M0, M1, M2, n=forget_retained['M0'],forget_retained['M1'],forget_retained['M2'], forget_retained['n_unlearned']
    ####################################################################
    # Figure setup
    ####################################################################
    figsize, lcols=fig_dict['figsize'],fig_dict['lcolor']
    fs, lw, mew, mfc, ms=fig_dict['fontsize'],fig_dict['linewidth'], fig_dict['mew'], fig_dict['mfc'],fig_dict['ms']
    marker, ls, lcolor, alpha=fig_dict['marker'],fig_dict['linestyle'],fig_dict['lcolor'], fig_dict['alpha']
    legend_loc, legend_ncols=fig_dict['legend_loc'],fig_dict['legend_ncols']
    legend_propsize, legend_box=fig_dict['legend_propsize'], fig_dict['legend_box']
    ylims=[np.ceil(np.min(M0.prototypes_))-1, np.ceil(np.max(M0.prototypes_))]
    ##########################
    # Xticklabels setup
    ###########################
    xticklabels=["" for x in range(len(feature_names))]
    for i, term in enumerate(Xtrain.columns):
        temp=term.split('_')
        if temp[0]=='num':
            xticklabels[i]=' '.join(temp[1:])
        else:
            xticklabels[i]=' '.join(temp)
    ###################################################################
    # Legend setup
    m1, m2, m3= mpatches.Patch(color=lcols[0],label='M0'), mpatches.Patch(color=lcols[1], label='M1'), mpatches.Patch(color=lcols[2], label='M2')
    handleorder1=[m1, m2, m3]
    ####################################################################
    # Plotting the bar plots representign prototypes
    ####################################################################
    fig5, ax5= plt.subplots(nrows=M0.prototypes_.shape[0], sharex=True, figsize=(figsize[0],figsize[1]), layout="constrained", dpi=100)
    for i in range(0, M0.prototypes_.shape[0]):
        prot_group=pd.DataFrame(data=np.array([feature_names,M0.prototypes_[i].T, M1.prototypes_[i].T, M2.prototypes_[i].T]).T,
                               columns=['features','M0','M1', 'M2'])#features)
        prot_group.set_index("features", inplace=True)
        prot_group[['M0','M1', 'M2']]=prot_group[['M0','M1', 'M2']].astype('float64')#.transform(np.exp)
        #prot_group[['M0','M1', 'M2']]=(prot_group[['M0','M1', 'M2']]-prot_group[['M0','M1', 'M2']].mean())/prot_group[['M0','M1', 'M2']].std()
        prot_group.plot.bar(ax=ax5[i], color=lcols, edgecolor=['k','k','k'])
        ax5[i].axhline(y=0.0, c="blue",linewidth=lw,zorder=0)
        ax5[i].set_ylabel('P-%d'%(i+1), fontsize=fs) 
        ax5[i].set_yticks(np.arange(ylims[0], ylims[1]+1, 2.0))
        ax5[i].tick_params(axis='both', which='minor', labelsize=fs-3)
        ax5[i].yaxis.set_tick_params(labelsize=fs-3)
        ax5[i].set_xticklabels(xticklabels, fontsize=fs)
        ax5[i].get_legend().remove()
        ax5[i].grid(which='both', linestyle = ls, linewidth = lw)
    ax5[i].set_xlabel('Features (log-transformed)', fontsize=fs)
    if ppc==1:
        fig5.legend(handles=handleorder1, loc=legend_loc, ncols=legend_ncols, prop={"size":legend_propsize},
               bbox_to_anchor=(legend_box[0], legend_box[1]))
    else:
        fig5.legend(handles=handleorder1, loc=legend_loc, ncols=legend_ncols, prop={"size":legend_propsize},
               bbox_to_anchor=(legend_box[0], legend_box[1]))
    fig5.suptitle('Original, retrained and unlearned GLVQ models on %s dataset'%dname, fontsize=fs)
    if 'suffix' in fig_dict:
        filename='%s%s_devProtsM0M1M2_swish_%s_nprot%d_%s.pdf'%(figpath,dname, solver_type, ppc, fig_dict['suffix'])
    else:
        filename='%s%s_%s_devProtsM0M1M2_swish_%s_nprot%d_samples%d.pdf'%(figpath,dname, exp_type, solver_type, ppc,n)
    fig5.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0.01)
            
    
def pca_scatterplot(dname:str, model_dict:dict, fig_dict:dict):
    """
    Arguments: 
    dname: name of the dataset 
    model_dict: dictionary containing the iteration number (iter), number of unlearned samples (nopt),
                the number of prototype per class (ppc), and the experiment type (exp_type)
    fig_dict: dictionary containing figure size, markers, marker size, marker face and edge colors, linestyle,
            linewidth, alpha, and legend details such as the prop size, location, legend anchoring position.
    What it does: Computes PCA based on the retrained set, and transforms the prototypes, unlearned samples (and 
    enforced samples, based on exp_type) accordingly.
    Returns:
        Plots the distribution of the samples from remembered set (retained/ retraining set), the unlearned 
        samples (forget set) (and the enforced set if the experiment type if enforced), and the prototypes from 
        the original (M0), retrained (M1), and unlearned (M2)  models. 

    """
    # Set up marker and colours
    ###############################################
    fs,linestyle, figsize= fig_dict['fontsize'],fig_dict['linestyle'], fig_dict['figsize']
    colors1, colors2, colors3= fig_dict['colors1'], fig_dict['colors2'], fig_dict['colors3']
    colors=[colors2, colors3]
    marker, ms, alpha, lw=fig_dict['marker'],fig_dict['markersize'],fig_dict['alpha'], fig_dict['lw']
    mew, mfc=fig_dict['mew'], fig_dict['mfc']
    # Set up legend and fig properties
    ###############################################
    legend_loc, legend_ncols=fig_dict['legend_loc'],fig_dict['legend_ncols']
    legend_propsize, legend_box=fig_dict['legend_propsize'], fig_dict['legend_box']
    ##################################################################
    retain_forget_data, retain_forget_label=import_forget_retained_sets(dname, model_dict)
    ppc, exp_type=model_dict['ppc'], model_dict['exp_type']
    lab_val=model_dict['lab_vals']
    forget_retained=import_model_utils(dname,model_dict)  
    M0, M1, M2, n=forget_retained['M0'],forget_retained['M1'],forget_retained['M2'], forget_retained['n_unlearned']
    zXretained, zXunlearn=retain_forget_data['retained'],retain_forget_data['unlearned']
    Yretained, Yunlearn=retain_forget_label['retained'],retain_forget_label['unlearned']
    #if ppc==1:
    pca_X=PCA(n_components='mle', svd_solver="full").fit(zXretained)
    pc_retrain_df,pc_unlearn_df=pca_X.transform(zXretained), pca_X.transform(zXunlearn)
    x_d , y_d,ux_d , uy_d= pc_retrain_df[:, 0], pc_retrain_df[:, 1], pc_unlearn_df[:, 0], pc_unlearn_df[:, 1]
    fig, ax= plt.subplots(nrows=1,figsize=(figsize[0],figsize[1]), dpi=100, layout="constrained")
    fig.suptitle("Discriminative projection GLVQ prototypes (for %s dataset)\n (PCA in log transformed space)"%dname, fontsize=fs+1)
    for i, cls in enumerate(M0.classes_):
        ii = cls == Yretained
        scatter=ax.scatter(x_d[ii],y_d[ii],c="white", marker=marker[0],s=ms[0],alpha=0.4,edgecolors=colors1[i],label=lab_val[i])
    for i, cls in enumerate(M0.classes_):
        ii = cls == Yunlearn
        scat1=ax.scatter(ux_d[ii],uy_d[ii],c=colors1[i],marker=marker[1], s=ms[1],edgecolors="k",label=lab_val[i], alpha=0.6)
    prot_group={}
    for cls in M0.classes_:
        for idx, labs in enumerate(M0.prototypes_labels_):
            if labs==cls:
                temp_prot_df=pd.DataFrame(data=np.array([M0.prototypes_[idx], M1.prototypes_[idx], M2.prototypes_[idx]]), 
                           columns=zXretained.columns)
                pc_prots_df=pca_X.transform(temp_prot_df)
                x_prot,y_prot =pc_prots_df[:, 0], pc_prots_df[:, 1]
                scat2=ax.scatter(x_prot,y_prot, c=colors[cls], marker=marker[2],s=ms[2], alpha=alpha, edgecolors="black",
                                 linewidth=lw, label=['M0-P%d'%cls,'M1-P%d'%cls,'M2-P%d'%cls])
    ax.set_xlabel("First eigenvector", fontsize=fs)
    ax.set_ylabel("Second eigenvector", fontsize=fs)     
    patch1,patch2 = mpatches.Patch(color='b', label=lab_val[0]), mpatches.Patch(color='r', label=lab_val[1])
    patch3,patch4 = mpatches.Patch(color=colors2[0], label='M0-P1'), mpatches.Patch(color=colors3[0], label='M0-P2')
    patch5,patch6 = mpatches.Patch(color=colors2[1], label='M1-P1'), mpatches.Patch(color=colors3[1], label='M1-P2')
    patch7,patch8 = mpatches.Patch(color=colors2[2], label='M2-P1'), mpatches.Patch(color=colors3[2], label='M2-P2')
    handleorder1=[patch1, patch3, patch5, patch7, patch2, patch4, patch6, patch8]
    fig.legend(handles=handleorder1, loc=legend_loc, ncols=legend_ncols, prop={"size":legend_propsize},
               bbox_to_anchor=(legend_box[0], legend_box[1]))
    if 'suffix' in fig_dict:
        filename='%s%s_Df_%s_ProtsM0M1M2_swish_%s_nprot%d_samples%d_%s.pdf'%(figpath, dname,exp_type, solver_type, ppc, n,
                                                                                   fig_dict['suffix'])
    else: #'%s%s_Df_enforced_ProtsM0M1M2_swish_%s_nprot%d.pdf'%(figpath, dname, solver_type, ppc)
        filename='%s%s_Df_%s_ProtsM0M1M2_swish_%s_nprot%d_samples%d.pdf'%(figpath, dname, exp_type,solver_type, ppc,n)
    fig.savefig(filename,dpi=100, bbox_inches="tight", pad_inches=0.01)
    
    
        