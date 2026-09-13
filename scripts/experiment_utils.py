import os, sys
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.metrics import roc_auc_score, f1_score
import random
from datetime import date
import time
from sklvq import GLVQ
import sklvq

parts=os.getcwd().split('/')
if parts[-1]=='notebooks':
    parts=parts[:-1]
print('/'.join(parts))
datapath='/'.join(parts)+'/data/'

code_path='/'.join(parts)+'/scripts/'
sys.path.append(code_path)


#normalization of original data
def dataset_health(dname):
    trainset, testset=pd.read_csv(datapath+'%s/%s_trainset.csv'%(dname, dname)), pd.read_csv(datapath+'%s/%s_testset.csv'%(dname,dname))
   # Xtrain=Xtrain.astype(float)#[Xtrain.columns[Xtrain.std()>0.05]].copy()
   # Xtest=Xtest.astype(float)
    #features=Xtrain.columns
    if dname=='criteo':
        exclude_ftrs=['conversion', 'visit', 'subset', 'exposure', 'shuff_group']
        features=list(np.setdiff1d(list(trainset.columns), exclude_ftrs))
        Ytrain, Ytest=trainset['visit'], testset['visit']
    else:
        features=trainset.columns[:-1]
        Ytrain, Ytest=trainset['Label'].to_numpy(), testset['Label'].to_numpy()
    Xtrain, Xtest=trainset[features], testset[features]
    return Xtrain, Ytrain, Xtest, Ytest, features

def data_normalization(Xtrain, Xtest):    
    mu, std=Xtrain.mean(skipna=True), Xtrain.std(skipna=True)
    zXtrain, zXtest=(Xtrain-mu)/std, (Xtest-mu)/std
    return zXtrain, zXtest


def data_norm_log(Xtrain, Xtest):    
    Xtrain, Xtest=Xtrain.astype(float), Xtest.astype(float)
    Xtrain[Xtrain<=0.0001]=0.0001
    Xtest[Xtest<=0.0001]=0.0001
    zXtrain, zXtest=np.log(Xtrain), np.log(Xtest)
    return zXtrain, zXtest
