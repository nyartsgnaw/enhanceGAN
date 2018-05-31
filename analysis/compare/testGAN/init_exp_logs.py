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

import pandas as pd
def init_exp_logs_df(model_type,tester_mode):
    exp_logs = pd.read_excel(os.path.join(CWDIR,'./../../../exp_logs.xlsx'))
    test_id = 0
    df = pd.DataFrame()
    for i in range(exp_logs.shape[0]):
        exp = exp_logs.iloc[i]
        exp_id = exp['exp_id']
        test_ls = [k for k,v in test2train_dict.items() if exp['trainData_id'] == v]
        equilibrium_avg,accs_false_avg,accs_true_avg = exp_logs.loc[exp_logs['exp_id']==exp_id,['equilibrium_avg','accs_false_avg','accs_true_avg']].values[0]
        for testData_id in sorted(test_ls):
            row = pd.DataFrame([test_id,exp_id,model_type,tester_mode,testData_id,test2train_dict[testData_id],None,None,None,None,None,None,None,None,
                                equilibrium_avg,accs_false_avg,accs_true_avg,
                                None,None,0],
                                index=['test_id', 'exp_id', 'model_type', 'tester_mode', 'testData_id','trainData_id',
                                            'AUC-test', 'rho-test', 'pValue-test', 'acc-test', 'AUC-val', 'rho-val',
                                            'pValue-val', 'acc-val', 
                                            'equilibrium_avg','accs_false_avg','accs_true_avg',
                                            'start_time', 'end_time', 'n_tested'])
            df = df.append(row.transpose())
            test_id+=1
    return df




def init_exp_logs_df_blank(model_type,tester_mode):
    exp_logs = pd.read_excel(os.path.join(CWDIR,'./../../../exp_logs.xlsx'))
    test_id = 0
    df = pd.DataFrame()
    for testData_id in test2train_dict.keys():
        trainData_id = test2train_dict[testData_id]
        exp_id = trainData_id        
        row = pd.DataFrame([test_id,exp_id,model_type,tester_mode,testData_id,test2train_dict[testData_id],None,None,None,None,None,None,None,None,
                            None,None,0],
                    index=['test_id', 'exp_id', 'model_type', 'tester_mode', 'testData_id','trainData_id',
                                'AUC-test', 'rho-test', 'pValue-test', 'acc-test', 'AUC-val', 'rho-val',
                                'pValue-val', 'acc-val', 
                                'start_time', 'end_time', 'n_tested'])
        df = df.append(row.transpose())
        test_id+=1
    return df




if __name__ == '__main__':
    init_exp_logs_df('G','0').to_excel(os.path.join(CWDIR,'./exp_logs_TG0.xlsx'),index=False)
    init_exp_logs_df_blank('B','0').to_excel(os.path.join(CWDIR,'./exp_logs_TB0.xlsx'),index=False)
    init_exp_logs_df_blank('B','1').to_excel(os.path.join(CWDIR,'./exp_logs_TB1.xlsx'),index=False)
    init_exp_logs_df('G','1').to_excel(os.path.join(CWDIR,'./exp_logs_TG1.xlsx'),index=False)
    init_exp_logs_df('D','2').to_excel(os.path.join(CWDIR,'./exp_logs_TD2.xlsx'),index=False)
#    init_exp_logs_df('G','2').to_excel(os.path.join(CWDIR,'./exp_logs_TG2.xlsx'),index=False)


