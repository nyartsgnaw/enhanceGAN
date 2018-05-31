import pandas as pd

import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()
try:
    with open(os.path.join(CWDIR,'./../../../data/.test2train_dict')) as f:
        raw_txt = f.read()
        test2train_dict = dict(zip([int(x) for x in raw_txt.split('\n')[0].split(',')], [int(x) for x in raw_txt.split('\n')[1].split(',')]))
except Exception as e:
    print(e)
    print('Please initialize the dictionary from testData_id to trainData_id at ~/enhanceGAN/data/.test2train_dict  ,')
    print('where the 1st row is testData_id split with comma, and 2nd row is trainData_id split with comma.')

def get_compare_df(df_raw):
    if pd.read_excel(os.path.join(CWDIR,'./exp_logs_TB{}.xlsx'.format(0))).isna().sum()['start_time']!=0:
        df_TB = pd.read_excel(os.path.join(CWDIR,'./exp_logs_TB{}.xlsx'.format(1)))
    
    elif pd.read_excel(os.path.join(CWDIR,'./exp_logs_TB{}.xlsx'.format(0))).isna().sum()['start_time']==0:
        df_TB = pd.read_excel(os.path.join(CWDIR,'./exp_logs_TB{}.xlsx'.format(0)))
    else:
        print('You need to run ~/enhanceGAN/analysis/compare/testGAN/run_testGAN -f exp_logs_TB0.xlsx before you compare the results.')
        return 
        
    if pd.read_excel(os.path.join(CWDIR,'./exp_logs_TG{}.xlsx'.format(0))).isna().sum()['start_time']!=0:
        df_TG = pd.read_excel(os.path.join(CWDIR,'./exp_logs_TG{}.xlsx'.format(1)))
    elif pd.read_excel(os.path.join(CWDIR,'./exp_logs_TG{}.xlsx'.format(0))).isna().sum()['start_time']==0:
        df_TG = pd.read_excel(os.path.join(CWDIR,'./exp_logs_TG{}.xlsx'.format(0)))
    else:
        print('You need to run ~/enhanceGAN/analysis/compare/testGAN/run_testGAN -f exp_logs_TG0.xlsx before you compare the results.')
        return         
        
    if pd.read_excel(os.path.join(CWDIR,'./exp_logs_TD{}.xlsx'.format(2))).isna().sum()['start_time']==0:
        df_TD = pd.read_excel(os.path.join(CWDIR,'./exp_logs_TD{}.xlsx'.format(2)))
    else:
        print('You need to run ~/enhanceGAN/analysis/compare/testGAN/run_testGAN -f exp_logs_TD2.xlsx before you compare the results.')
        return                 

    df_exp_logs = pd.read_excel(os.path.join(CWDIR,'./../../../exp_logs.xlsx'))

    df = pd.DataFrame()
    names = ['AUC-test', 'rho-test', 'pValue-test', 'acc-test', 'AUC-val', 'rho-val','pValue-val', 'acc-val' ]
    for i in range(df_raw.shape[0]):
        try:
            exp_id = df_raw['exp_id'].iloc[i]
            testData_id = df_raw['testData_id'].iloc[i]

            setup_id, model_id = df_exp_logs.loc[df_exp_logs['exp_id']==exp_id,['setup_id','model_id']].values.tolist()[0]

            row_TD = df_TD.loc[(df_TD['exp_id']==exp_id)&(df_TD['testData_id']==testData_id),:].iloc[0]
            row_TG = df_TG.loc[(df_TG['exp_id']==exp_id)&(df_TG['testData_id']==testData_id),:].iloc[0]
            row_TB = df_TB.loc[df_TB['testData_id']==testData_id,:].iloc[0]
            row = pd.DataFrame([exp_id,testData_id,test2train_dict[testData_id],setup_id,model_id]\
                                +row_TD[names].tolist()\
                                +row_TG[names].tolist()\
                                +row_TB[names].tolist()\
                                +row_TD[['equilibrium_avg','accs_false_avg','accs_true_avg']].tolist()
                        ,index=['exp_id', 'testData_id','trainData_id','setup_id','model_id']\
                            +[x+'-D' for x in names]\
                            +[x+'-G' for x in names]\
                            +[x+'-B' for x in names]\
                            +['equilibrium_avg','accs_false_avg','accs_true_avg'])
            df = df.append(row.transpose())
        except Exception as e:
            print(e)
            df = df.append(pd.DataFrame())
    df = df.sortlevel(axis=1)
    return df 

if __name__ == '__main__':
#    df_TG_best = get_compare_df(pd.read_csv('./../output/best_exp_dict_TG.csv'))
#    df_TG_best.to_excel('./../output/df_compare_TD.xlsx',index=False)
#    df_TD_best = get_compare_df(pd.read_csv('./../output/best_exp_dict_TD.csv'))
#    df_TD_best.to_excel('./../output/df_compare_TG.xlsx',index=False)


    df_TG = get_compare_df(pd.read_excel(os.path.join(CWDIR,'./exp_logs_TG0.xlsx')))
    df_TG.to_excel(os.path.join(CWDIR,'./../output/df_compare.xlsx'),index=False)



    

                            
    
                            