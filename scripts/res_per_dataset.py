import os, os
import pandas as pd
parts=os.getcwd().split('/')
print(parts)
if parts[-1]=='scripts':
    parts=parts[:-1]
srcpath='/'.join(parts)+'/results/'

solver_types=['sgd']
dname_all=['breastcancer','diabetes', 'surgical', 'banking', 'adult', 'criteo']

def combine_tabs_per_dataset(dname, exp_type):
    c=0
    for solver_type in solver_types:
        for nprots_per_class in [1,2]:#,3]:
          #  if dname=='criteo':
            tab_filename='%s%s/%s_%s_swish_%s_nprot%d0.csv'%(srcpath, dname, dname, exp_type, solver_type, nprots_per_class)
            print('filename: ',tab_filename)
            #else:
            #tab_filename='%s%s/%s_%s_unlearn_enforce_%s_nprot%d.csv'%(srcpath, dname, dname, exp_type,solver_type, nprots_per_class)
            if os.path.exists(tab_filename)==False:
                print(tab_filename, '\nfile does not exist')
                continue
            else:
                print(tab_filename)
                print('file exists')
     #   if os.path.exists(tab_filename):      
                resdf1=pd.read_csv(tab_filename, sep='\t')
                if c==0:
                    rescols=list(resdf1.columns)
                    resdf1['solver_type']=solver_type
                    resdf=resdf1.copy()
                    c+=1
                else:
                    resdf1['solver_type']=solver_type
                    resdf=pd.concat([resdf, resdf1])
                    c+=1                 
    if c>0:# 'resdf' in locals():
        resdf=resdf[['solver_type']+rescols].copy()
        if 'Mapping' in resdf:
            resdf.drop(['Mapping'],axis=1, inplace=True)
        resdf.rename(columns={'n':'num_unlearned_samples', 'num_prot': 'prot_per_class'}, inplace=True)
        print(resdf.head(5))
        respath='%s/%s_%s_unlearning_0.csv'%(srcpath, dname, exp_type)
        #respath='%s/%s_%s_unlearning_0.csv'%(srcpath, dname, exp_type)
        resdf.to_csv(respath, sep='\t', index=False)
        return resdf
    else:
        print('Nothing to return for %s %s'%(exp_type, dname))
  #      resdf=0
  #      return resdf


def loop_across_datasets():
    exp_types=['random', 'outlier']
    for dname in dname_all:
        resdf_random=combine_tabs_per_dataset(dname, exp_types[0])
        print(type(resdf_random))
       # if resdf_random!=0:
            #print(dname, resdf_random.shape)
        print(dname, resdf_random.shape)
        if (dname=='breastcancer'):
            continue
        else:
            resdf_outlier=combine_tabs_per_dataset(dname, exp_types[1])
            print(type(resdf_outlier))
            print(dname, resdf_outlier.shape)


loop_across_datasets()
