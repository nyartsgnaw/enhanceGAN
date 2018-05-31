#this file is used for finding best experiment by the discriminator AUC on validation dataset
#the highest experiment will be anchored with corresponding training data id, and saved to ./../output/best_exp_dict.csv

import pandas as pd
import numpy as np 
import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()

def rank_exp_by(test_name,rank_feature='AUC-val'):

    for tester_mode in [0,1,2]:
        try:
            names = ['AUC','rho','pValue','acc']
            df_test_log = pd.read_excel(os.path.join(CWDIR,'./../testGAN/exp_logs_{}.xlsx'.format(test_name+str(tester_mode))))
            rows = [x[1].sort_values(rank_feature,ascending=False).iloc[0] for x in df_test_log.groupby('testData_id')]
            df_best = pd.DataFrame(rows)
            df_exp_logs = pd.read_excel(os.path.join(CWDIR,'./../../../exp_logs.xlsx'))
            df_output = pd.DataFrame()
            for i in range(df_best.shape[0]):
                row = df_best.iloc[i]
                train_info = df_exp_logs.loc[df_exp_logs['exp_id']==row['exp_id'],['trainData_id','setup_id','model_id']].iloc[0]
                row_comb = pd.concat([train_info,
                    row[['exp_id','testData_id']\
                    +[x+'-test' for x in names]\
                    +[x+'-val' for x in names]
                ]])
                df_output=  df_output.append(row_comb,ignore_index=True)
            
            os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./../output/')))
            df_output.to_csv(os.path.join(CWDIR,'./../output/best_exp_dict_{}.csv'.format(test_name)),index=False)
        except Exception as e:
            print(e)     

    return df_output         

if __name__ == '__main__':
    rank_exp_by(test_name='TD',rank_feature='AUC-val')
    rank_exp_by(test_name='TG',rank_feature='AUC-val')
               
