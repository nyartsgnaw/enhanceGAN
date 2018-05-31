import numpy as np
import pandas as pd
import os
import re 
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()
def import_local_package(addr_pkg,function_list=[]):
	import importlib.util
	spec = importlib.util.spec_from_file_location('pkg', addr_pkg)
	myModule = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(myModule)
	if len(function_list)==0:
		import re
		function_list = [re.search('^[a-zA-Z]*.*',x).group() for x in dir(myModule) if re.search('^[a-zA-Z]',x) != None]

	for _f in function_list:
		try:
			eval(_f)
		except NameError:
			exec("global {}; {} = getattr(myModule,'{}')".format(_f,_f,_f)) #exec in function has to use global in 1 line
			print("{} imported".format(_f))

	return
import_local_package(os.path.join(CWDIR,'./../../lib/utils/utils.py'),['get_batch_idx'])


if __name__ == '__main__':
	try:
		CWDIR = os.path.abspath(os.path.dirname(__file__))
	except:
		CWDIR = os.getcwd()

	CV_ratio = {'train':9,'val':1,'test':0}

	df_train = pd.read_csv(os.path.join(CWDIR,'./../CDK/df_train.csv'))
	groups = df_train.groupby(['type'])


	indexes_dict = {k:list(range(len(groups.indices[k]))) for k,v in groups.indices.items()}
	b_idx_dict = {k:get_batch_idx(indexes_dict[k],sum(list(CV_ratio.values()))) for k,v in indexes_dict.items()}


	idx_train = {k:groups.indices[k][sum(b_idx_dict[k][0:(CV_ratio['train'])],[])] for k,v in list(groups)}
	idx_val = {k:groups.indices[k][sum(b_idx_dict[k][(CV_ratio['train']):(CV_ratio['train']+CV_ratio['val'])],[])] for k,v in list(groups)}
	idx_test = {k:groups.indices[k][sum(b_idx_dict[k][(CV_ratio['train']+CV_ratio['val']):(CV_ratio['train']+CV_ratio['val']+CV_ratio['test'])],[])] for k,v in list(groups)}

	dfs_train = {k:df_train.iloc[idx_train[k]] for k,v in idx_train.items()}
	dfs_val = {k:df_train.iloc[idx_val[k]] for k,v in idx_val.items()}
	dfs_test = {k:df_train.iloc[idx_test[k]] for k,v in idx_test.items()}

	os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./../../data/testData/')))
	os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./../../data/realTestData/')))
	os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./../../data/valData/')))
	os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./../../data/trainData/')))
	i = 0 
	while i < len(sorted(dfs_train)):
		k = sorted(dfs_train)[i]
		#get data
		dfs_test[k].index.name = 'original_index'
		(dfs_test[k]).to_csv(os.path.join(CWDIR,'./../../data/testData/testData_rawid{}.csv').format(i),index=True)
		dfs_val[k].index.name = 'original_index'
		(dfs_val[k]).to_csv(os.path.join(CWDIR,'./../../data/valData/valData_rawid{}.csv').format(i),index=True)
		dfs_train[k].index.name = 'original_index'
		(dfs_train[k]).to_csv(os.path.join(CWDIR,'./../../data/trainData/trainData_rawid{}.csv').format(i),index=True)
		i+=1

	df_test = pd.read_csv('./../CDK/df_test.csv')
	for k,v in df_test.groupby(['partition_id']).groups.items():
		df_t = df_test.iloc[v]
		df_t.to_csv(os.path.join(CWDIR,'./../../data/realTestData/testData_rawid{}.csv').format(i),index=True)
		i+=1
	menu_cols = ['testData_id','type','n_samples','pos_prop','description','addr']
	df_menuVal = pd.DataFrame(columns = menu_cols)
	df_menuTest = pd.DataFrame(columns = menu_cols)
	df_menuTrain = pd.DataFrame(columns = menu_cols)

	#get only these happend in sNebula test dataset


	j = 0
	i = 0
	while i <len(sorted(dfs_train)):
		k = sorted(dfs_train)[i]
		CDK = k
		n_val,n_test,n_train = int(dfs_val[k].shape[0]),int(dfs_test[k].shape[0]),int(dfs_train[k].shape[0])
#		if (n_val >200)|(k in keys_sNebula):
		if True:
			addr_val = os.path.abspath(os.path.join(CWDIR,'./../../data/valData/valData_rawid{}.csv').format(i))
			addr_val = '.'+re.search('(?<=enhanceGAN).*',addr_val).group()
			addr_test = os.path.abspath(os.path.join(CWDIR,'./../../data/testData/testData_rawid{}.csv').format(i))
			addr_test = '.'+re.search('(?<=enhanceGAN).*',addr_test).group()
			addr_train = os.path.abspath(os.path.join(CWDIR,'./../../data/trainData/trainData_rawid{}.csv').format(i))
			addr_train = '.'+re.search('(?<=enhanceGAN).*',addr_train).group()
#			print(i,n_val,n_test,n_train,k,addr_test)
			
			#update menu
			df_menuVal.loc[j,:] = [j,type,n_val,dfs_val[k].loc[dfs_val[k]['label']==1,].shape[0]/(n_val+1),'',addr_val]

			df_menuTest.loc[j,:] = [j,CDK,n_test,dfs_test[k].loc[dfs_test[k]['label']==1,].shape[0]/(n_test+1),'',addr_test]

			df_menuTrain.loc[j,:] = [j,CDK,n_train,dfs_train[k].loc[dfs_train[k]['label']==1,].shape[0]/(n_train+1),'',addr_train]
			j+=1
		i+=1	
	dict_test_idx = df_test.groupby(['type','partition_id']).groups
	for k,v in dict_test_idx.items():
		CDK,partition_id = k
		if (CDK) in keys_sNebula:
			print(i,k,addr_test)
			addr_realTest = os.path.abspath(os.path.join(CWDIR,'./../../data/realTestData/testData_rawid{}.csv').format(i))
			addr_realTest = '.'+re.search('(?<=enhanceGAN).*',addr_realTest).group()
			n_test = df_test.iloc[dict_test_idx[k]].shape[0]
			df_menuTest.loc[j,:] = [j,CDK,n_test,\
									df_test.iloc[dict_test_idx[k]].loc[df_test['label']==1,].shape[0]/n_test,\
									'',addr_realTest]

			df_t = df_test.iloc[v]
			df_t.to_csv(os.path.join(CWDIR,'./../../data/realTestData/testData_rawid{}.csv').format(i),index=True)
			i+=1
			j+=1

	


	df_menuVal.to_csv(os.path.join(CWDIR,'./../../experiments/menu_valData.csv'),index=False)
	df_menuTest.to_csv(os.path.join(CWDIR,'./../../experiments/menu_testData.csv'),index=False)
	df_menuTrain.to_csv(os.path.join(CWDIR,'./../../experiments/menu_trainData.csv'),index=False)






	










