import datetime
import os
import sys
import pandas as pd
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()

#sys.path.append(os.path.join(CWDIR,'./../../lib/utils/'))

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
	
import_local_package(os.path.join(CWDIR,'./data/lib/prepare_data.py'),['prepare_data'])
import_local_package(os.path.join(CWDIR,'./lib/utils/train_GAN.py'),['train_GAN'])
import_local_package(os.path.join(CWDIR,'./lib/utils/utils.py'),['test_model','train_model','save_df'])

from keras import metrics
from keras.optimizers import SGD, Adam, RMSprop
pd.options.mode.chained_assignment = None  # default='warn'




def start_GAN(exp):
	start_time = datetime.datetime.now()
	df_setup= pd.read_csv(os.path.join(CWDIR,'./experiments/menu_setup.csv'))
	setup_info = df_setup.loc[df_setup['setup_id']==exp['setup_id'],].iloc[0]


	pdata=prepare_data(X_encode = setup_info['X_encode'],Y_encode = 'integer')
	_,_,X_val,Y_val,X_test,Y_test = pdata.get_data(exp['trainData_id'],exp['valData_id'],1)	


	pdata=prepare_data(X_encode = setup_info['X_encode'],Y_encode = setup_info['Y_encode'])
	X_train,Y_train,_,_,_,_ = pdata.get_data(exp['trainData_id'],exp['valData_id'],1)	

	length = X_train.shape[1]


#	adam0=Adam(lr=0.03, beta_1=0.9 ,decay=0.001)
#	rmsprop0 = RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.01)
	sgd1_G = SGD(lr=setup_info['lr_G'], decay=0.01, momentum=0.9, nesterov=True)
	sgd1_D = SGD(lr=setup_info['lr_D'], decay=0.01, momentum=0.9, nesterov=True)
	adam1_G=Adam(lr=0.01, beta_1=0.9 ,decay=0.001)
	adam1_D=Adam(lr=0.01, beta_1=0.9 ,decay=0.001)
	rmsprop1_G = RMSprop(lr=setup_info['lr_G'], rho=0.9, epsilon=1e-08, decay=0.001)
	rmsprop1_D = RMSprop(lr=setup_info['lr_D'], rho=0.9, epsilon=1e-08, decay=0.001)
	rmsprop0 = RMSprop(lr=0.004, rho=0.9, epsilon=1e-08, decay=0.01)


	if setup_info['opt_G'].lower() == 'sgd':
		opt_G = sgd1_G
	elif setup_info['opt_G'].lower() == 'adam':
		opt_G = adam1_G
	elif setup_info['opt_G'].lower() == 'rmsprop':
		opt_G = rmsprop1_G
	if setup_info['opt_D'].lower() == 'sgd':
		opt_D = sgd1_D
	elif setup_info['opt_D'].lower() == 'adam':
		opt_D = adam1_D
	elif setup_info['opt_D'].lower() == 'rmsprop':
		opt_D = rmsprop1_D

#	with open('./lib/.model_id','w') as f:
#		f.write(exp['model_id'])

	import importlib.util
	spec = importlib.util.spec_from_file_location(exp['model_id'], os.path.join(CWDIR,"./experiments/models/{}.py").format(exp['model_id']))
	myModel = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(myModel)
	create_G,create_D,create_T = myModel.create_G,myModel.create_D,myModel.create_T
	G = create_G(input_dim =(100,),output_dim= X_train[0].shape)
	D = create_D(input_dim =X_train[0].shape,output_dim=3 )
	G.compile(loss='categorical_crossentropy', optimizer = adam1_G, metrics=[metrics.categorical_accuracy])
	D.compile(loss='categorical_crossentropy', optimizer = adam1_D, metrics=[metrics.categorical_accuracy])   #this will lead to poor performance of Discriminator at about epoch 39

	GAN = train_GAN(
				X_encode = setup_info['X_encode'],\
				Y_encode = setup_info['Y_encode'],\
				exp_id = str(exp['exp_id']),\
				D_class_weight={'true':1.2,'fake':1,'false':1.2},\
				G_class_weight={'true':1,'fake':1,'false':1},\
				noise_ratio=setup_info['noise_ratio'],\
				save_model=True,\
				is_generating_stochastic = setup_info['is_generating_stochastic'])

	if str(setup_info['is_tester']) == str(1):
		T = create_T(input_dim =X_train[0].shape,output_dim=1 )
		T.compile(loss='binary_crossentropy', optimizer = rmsprop0, metrics=['accuracy'])
	else:
		T = None
		X_val = None
		Y_val = None

	G,D = GAN.init_GD_models(G,D,X_train,Y_train,\
						n_epoch_init_G = int(setup_info['n_epoch_init_G']),
						n_epoch_init_D = int(setup_info['n_epoch_init_D']))


	G.compile(loss=setup_info['loss_G'], optimizer = opt_G)
	D.compile(loss=setup_info['loss_D'], optimizer = opt_D)   #this will lead to poor performance of Discriminator at about epoch 39
	#whether to test on a classifier or not


	output =GAN.train_GAN(
			X_train=X_train,
			Y_train=Y_train,
			X_val=X_val,
			Y_val=Y_val,
			G=G,
			D=D,
			T=T,
			batch_size = int(setup_info['batch_size']),\
			n_epoch=int(setup_info['n_epoch']),\
			n_epoch_T = int(setup_info['n_epoch_T']),\
			val_prop = setup_info['val_prop'],\
			patient=setup_info['patient'],\
			is_categorical_weight_control=setup_info['is_categorical_weight_control'],\
			is_monitored=setup_info['is_monitored'],\
			is_early_stopping=setup_info['is_early_stopping'])
	

	exp['start_time'] = str(start_time.replace(microsecond=0))
	exp['end_time'] = str(datetime.datetime.now().replace(microsecond=0))
	
	exp['accs_true_avg'] = output['records']['accs_true_avg']
	exp['accs_false_avg'] = output['records']['accs_false_avg']
	exp['equilibrium_avg'] = output['records']['equilibrium_avg']
	try:
		exp['generated_samples'] = output['records']['generated_samples']
		exp['i_epoch'] = output['records']['i_epoch']
		exp['AUC'] = output['records']['AUC']
		exp['rho'] = output['records']['rho']
		exp['pValue'] = output['records']['pValue']
		exp['acc'] = output['records']['acc']
	except:
		pass
	exp['paths'] = GAN.paths
	del G,D,output,X_train,Y_train,X_val,Y_val,GAN
	
	def save_model_paths(exp):
		import re
		addr_model = os.path.join(CWDIR,'./experiments/menu_model_addr.csv')
		paths = exp['paths']
		for k,v in paths.items():
			paths[k] = '.'+re.search('(?<=enhanceGAN).*',v).group()
		row_paths = pd.DataFrame([str(int(exp['exp_id'])),paths['G_path'],paths['D_path'],paths['T_path']],
							index=['exp_id','G_path','D_path','T_path']).transpose()

		save_df(addr_model,row_paths)
		del exp['paths']
		return 

	save_model_paths(exp)
	exp['n_tested']+=1

	return exp



if __name__ == '__main__':

	exp_addr = os.path.join(CWDIR,'./exp_logs.xlsx')
	df = pd.read_excel(exp_addr)
	for i in range(df.shape[0]):
		if df.iloc[i]['n_tested'] == 0:
			exp = df.iloc[i]
			print(exp)
			try:
				exp = start_GAN(exp)
			except Exception as e:
				print(e) 
			df.loc[i,:] = exp
#			df = df.sortlevel(axis=1)
			df.to_excel(exp_addr,index=False)

