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
    if dname=='adult':
        features=['age', 'num-sex', 'race-summary', 'education-num', 'num-marital',
           'fnlwgt', 'capital-gain', 'hours-per-week', 'Reg_US', 'Work_Private']
    elif dname=='diabetes':
        features=['num_gender','num_cat_age', 'num_change', 'num_diabetesMed', 'time_in_hospital',
           'num_lab_procedures', 'num_procedures', 'num_medications','number_outpatient', 
           'number_emergency', 'number_inpatient','number_diagnoses', 'num_acarbose', 'num_acetohexamide', 
           'num_insulin','num_chlorpropamide', 'num_citoglipton', 'num_examide', 'num_glimepiride',
           'num_pioglitazone','num_glipizide', 'num_metformin', 'num_glyburide',  'num_rosiglitazone',
           'num_glimepiride-pioglitazone', 'num_metformin-pioglitazone', 'num_glipizide-metformin', 
           'num_glyburide-metformin', 'num_metformin-rosiglitazone', 'num_miglitol', 'num_nateglinide', 
           'num_repaglinide', 'num_tolazamide', 'num_tolbutamide', 'num_troglitazone',
           'Disch_Home/hospice', 'AfricanAmerican', 'Caucasian']
    elif dname=='surgical':
        features=['bmi', 'Age','gender','race', 'asa_status', 'baseline_cancer', 'baseline_charlson',
           'baseline_cvd', 'baseline_dementia', 'baseline_diabetes','baseline_digestive', 'baseline_osteoart', 
           'baseline_psych','baseline_pulmonary', 'ahrq_ccs', 'ccsComplicationRate',#'ccsMort30Rate',
           'complication_rsi', 'mortality_rsi', #'dow', 'hour', 'month','moonphase', 'mort30', 
            ]
    return Xtrain, Ytrain, Xtest, Ytest, features

def data_normalization(Xtrain, Xtest):
    stol=10^(-5)
    mu, std=Xtrain.mean(skipna=True), Xtrain.std(skipna=True)
    if std<=stol:
        std=1
    zXtrain, zXtest=(Xtrain-mu)/std, (Xtest-mu)/std
    return zXtrain, zXtest


def data_norm_log(Xtrain, Xtest):    
    Xtrain, Xtest=Xtrain.astype(float), Xtest.astype(float)
    Xtrain[Xtrain<=0.0001]=0.0001
    Xtest[Xtest<=0.0001]=0.0001
    zXtrain, zXtest=np.log(Xtrain), np.log(Xtest)
    return zXtrain, zXtest
