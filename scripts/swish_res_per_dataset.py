import os, os
import pandas as pd
parts=os.getcwd().split('/')
print(parts)
if parts[-1]=='scripts':
    parts=parts[:-1]
srcpath='/'.join(parts)+'/results/'

solver_types=['sgd']
dname_all=['breastcancer','diabetes', 'surgical', 'banking', 'adult']#, 'criteo']

def combine_tabs_per_dataset(dname, exp_type):
    c=0
    for solver_type in solver_types:
        for nprots_per_class in [1,2]:
            #resultspath+'%s/%s_random_unlearn_swish_%s_nprot%d0.csv'%(dname, dname,solver_type, nprots_per_class)
            tab_filename='%s%s/%s_random_unlearn_swish_%s_nprot%d0.csv'%(srcpath, dname, dname,solver_type, nprots_per_class)
            print(tab_filename)
            if os.path.exists(tab_filename):
                print('file exists')
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
            else:
                print('file does not exist')
    resdf=resdf[['solver_type']+rescols].copy()
  #  resdf.drop(['Mapping'],axis=1, inplace=True)
    resdf.rename(columns={'n':'n_unlearned', 'num_prot': 'prot_per_class'}, inplace=True)
    print(resdf.head(5))
    respath='%s/%s_%s_unlearning_swish.csv'%(srcpath, dname, exp_type)
    resdf.to_csv(respath, sep='\t', index=False)
    return resdf


def loop_across_datasets():
    exp_types=['random', 'outlier']
    for dname in dname_all:
        resdf_random=combine_tabs_per_dataset(dname, exp_types[0])
        print(dname, resdf_random.shape)
        resdf_outlier=combine_tabs_per_dataset(dname, exp_types[1])
        print(dname, resdf_outlier.shape)


loop_across_datasets()
