import numpy as np
import pandas as pd
import os
if __name__ == '__main__':
	try:
		CWDIR = os.path.abspath(os.path.dirname(__file__))
	except:
		CWDIR = os.getcwd()
	CV_ratio = {'train':8,'val':1,'test':1}
	names_data = [os.path.join(CWDIR,'./../raw/',x) for x in os.listdir(os.path.join(CWDIR,'./../raw/'))]

	dfs = [pd.read_csv(x,delimiter='"',header=0) for x in names_data]
	legal_names = ['target_name ','canonical_smiles','ecfp','measurement_value','new_measurement_value','measurement_type','ligand_eff']

	for df in dfs:
		for col in df.columns:
			if col not in legal_names:
				del df[col]
				print('delete column:{}'.format(col))

	dfs_ls = []
#	dfs_dict = {}
	for df in dfs:
		df.columns = [x.strip().replace(' ','-') for x in df.columns]
#		key = (df['target_name'].iloc[0]).strip().replace(' ','-')
		tmp = df[['target_name','canonical_smiles','ecfp','new_measurement_value']]
		tmp['label'] = (tmp['new_measurement_value']<200).values.astype(int)
		del tmp['new_measurement_value']
#		dfs_dict[key] =tmp
		dfs_ls.append(tmp)
	df_train = pd.concat(dfs_ls,axis=0)
	df_train.columns=['type','canonical_smiles','ecfp','label']
	df_train['type'] = [x.replace('Cyclin-dependent kinase ','CDK').strip() for x in df_train['type'].values]
	df_train.to_csv(os.path.join(CWDIR,'./..','df_train.csv'),index=False)

	df_test = pd.DataFrame(columns=list(df_train.columns)+['partition_id'])
	df_test.to_csv(os.path.join(CWDIR,'./..','df_test.csv'),index=False)



