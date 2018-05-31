import datetime
import os
import sys
import numpy as np 
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()
from keras.optimizers import SGD, Adam, RMSprop



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

#import_local_package(os.path.join(CWDIR,'./../../../lib/utils/utils.py'),['load_model'])
from keras.models import  load_model
import_local_package(os.path.join(CWDIR,'./../../../data/lib/prepare_data.py'),['prepare_data'])
import_local_package(os.path.join(CWDIR,'./../../../lib/utils/train_GAN.py'),['train_GAN'])
import_local_package(os.path.join(CWDIR,'./../../../lib/utils/utils.py'),['test_model_D','test_model','train_model','save_df'])
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


		


class test_enhanceGAN(object):
	#define the class that test enhanceGAN
	#it integrates a bunch of existed the logics and avails ./exp_logs_T.xlsx. 
	# e.g. experiments logic in ./../../../exp_logs.xlsx,
	#logic in optimal models address and logs address,
	#and relationships between trainData_id,valData_id,testData_id,exp_id,setup_id.
	def __init__(self):
		try:
			CWDIR = os.path.abspath(os.path.dirname(__file__))
		except:
			CWDIR = os.getcwd()

		self.tester_modes = [0,1,2]
		dirs = ['trained_models','predictions','test_history','fakeData']
		for _d in dirs:
			for self.model_type in ['D','G','B']:
				for self.tester_mode in self.tester_modes:
					os.system('mkdir -p {}/'.format(os.path.join(CWDIR,'./logs/',_d)))
		
		self.exp_logs = pd.read_excel(os.path.join(CWDIR,'./../../../exp_logs.xlsx'))

		self.GAN = train_GAN(
					X_encode = 'onehot',\
					Y_encode = 'integer',\
					exp_id = '0',\
					D_class_weight={'true':1.2,'fake':1,'false':1.2},\
					G_class_weight={'true':1,'fake':1,'false':1},\
					is_generating_stochastic = False)

	def build_dataset(self,trainData_id,valData_id,testData_id):
		#get data
		X_train,Y_train,X_val,Y_val,X_test,Y_test = self.GAN.pdata.get_data(trainData_id,valData_id,testData_id)	
		# prepare training data


		if self.tester_mode == 0:
			if (self.model_type == 'B'):
				pass #no need to change anything, the original training data will work

			if (self.model_type == 'D'):
				print('Attempting to train D, this is basically retrain GAN, should call ~/enhanceGAN/lib/utils/train_GAN.py directly')


			elif (self.model_type == 'G'):

				train_data = self.GAN.init_GAN_trainingData(X_train=X_train,Y_train=Y_train)
				# get generator
				df_model_addr = pd.read_csv(os.path.join(CWDIR,'./../../../experiments/menu_model_addr.csv'))
				G_path = os.path.join(CWDIR,'../../../',df_model_addr.loc[df_model_addr['exp_id']==self.test_exp_id,'G_path'].values[-1])
				G = load_model(G_path)
				#get fake data
				G_samples = self.GAN.pdata.generate_data(G,train_data['X_noise'],X_encode='onehot',is_generating_stochastic=False)
				G_samples = np.array([self.GAN.pdata.encode_seq2mat(x) for x in set([self.GAN.pdata.decode_mat2seq(x) for x in G_samples]) ])
				train_data.update({'X_fake':G_samples,'Y_fake':np.array([np.array([0,1,0])]*G_samples.shape[0])})
				self.data_mix = self.GAN.mix_data(G,train_data)
				X_train = self.data_mix['X_train']
				Y_train = self.data_mix['Y_train']

		#prepare testing data
		if (self.model_type == 'B'):
			pass #no need to change anything, the original testing data will work

		if (self.model_type == 'G'):
			pass #no need to change anything, the original testing data will work

		elif (self.model_type == 'D'):
			test_data = self.GAN.init_GAN_trainingData(X_train=X_test,Y_train=Y_test)
			X_test = np.concatenate([test_data['X_true'],test_data['X_false']])
			Y_test = np.concatenate([test_data['Y_true'],test_data['Y_false']])
			val_data = self.GAN.init_GAN_trainingData(X_train=X_val,Y_train=Y_val)
			X_val = np.concatenate([val_data['X_true'],val_data['X_false']])
			Y_val = np.concatenate([val_data['Y_true'],val_data['Y_false']])
		
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test
		self.X_val = X_val
		self.Y_val = Y_val
		return X_train,Y_train,X_val,Y_val,X_test,Y_test

	def get_model(self):
		##get _model 
		## load
		if (self.tester_mode==2):
			if (self.model_type == 'B'):
				pass
			if (self.model_type == 'D'):
				model_load_addr = os.path.join(CWDIR,'./../../../logs/trained_models/GAN/GAN_D_exp{}.h5'.format(self.test_exp_id))
			elif (self.model_type == 'G'):
				model_load_addr = os.path.join(CWDIR,'./../../../logs/trained_models/tester/GAN_T_exp{}.h5'.format(self.test_exp_id))
		elif (self.tester_mode ==1):
			if (self.model_type == 'B'):
				model_load_addr = os.path.join(CWDIR,'./logs/trained_models/T{}_{}.h5'.format(self.model_type,self.test_exp_id))
			if (self.model_type == 'D'):
				pass
			elif (self.model_type == 'G'):
				model_load_addr = os.path.join(CWDIR,'./logs/trained_models/T{}_{}.h5'.format(self.model_type,self.test_exp_id))


		if (self.tester_mode==2)|(self.tester_mode==1):
			# get model
			model = load_model(model_load_addr)
		## train 
		if (self.tester_mode == 0):

			if (self.model_type == 'D'):
				print('Attempting to train D, this is basically retrain GAN, should call ~/enhanceGAN/lib/utils/train_GAN.py directly')
			else:
				model = create_T(input_dim=(self.X_train.shape[1],self.X_train.shape[2]),output_dim = 1)
				sgd1_G = SGD(lr=0.01, decay=0.01, momentum=0.9, nesterov=True)
				sgd1_D = SGD(lr=0.01, decay=0.01, momentum=0.9, nesterov=True)
				adam1_G=Adam(lr=0.01, beta_1=0.9 ,decay=0.001)
				adam1_D=Adam(lr=0.01, beta_1=0.9 ,decay=0.001)
				rmsprop1_G = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.001)
				rmsprop1_D = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.001)
				rmsprop0 = RMSprop(lr=0.004, rho=0.9, epsilon=1e-08, decay=0.01)
				model.compile(loss='binary_crossentropy', optimizer = rmsprop0, metrics=['accuracy'])			
				
				
				model = train_model(
					model=model,\
					X_train=self.X_train,\
					Y_train=self.Y_train,\
					n_epoch =200,\
					monitor='acc',\
					min_delta=0.001,\
					patience=20,\
					verbose=2,\
					mode='auto',\
					shuffle = True,\
					validation_split=0.2,\
					log_path = os.path.join(CWDIR,'./logs/test_history/T{}_log_{}.csv'.format(self.model_type,self.save_id)),\
					model_path = os.path.join(CWDIR,'./logs/trained_models/T{}_{}.h5'.format(self.model_type,self.save_id)))	
		
		self.model = model
		return model


	def test_model(self,X_test,Y_test,model):
		#test model on test
		if (self.model_type == 'G') | (self.model_type == 'B'):
			output = test_model(X_test = X_test,Y_test=Y_test,model=model,cutoff=0.5)
		elif (self.model_type == 'D'):
			output = test_model_D(X_test = X_test,Y_test=Y_test,model=model,cutoff=0.5)
		return output

	def start(self,exp):

		start_time = datetime.datetime.now()

		#read self.test_exp_id 
		self.test_exp_id = int(exp['exp_id'])
		model_id = str(self.exp_logs.loc[self.exp_logs['exp_id']==self.test_exp_id,:]['model_id'].values[0])
		import_local_package(os.path.join(CWDIR,'./../../../experiments/models/{}.py'.format(model_id)),['create_T','create_G','create_D'])
		self.tester_mode = int(exp['tester_mode'])
		self.model_type = str(exp['model_type'])

		if (self.model_type == 'G'):
			self.save_id = self.test_exp_id
		elif (self.model_type == 'B'):
			self.save_id = self.exp_logs.loc[self.exp_logs['exp_id']==self.test_exp_id,'trainData_id'].values[0]

		trainData_id = int(self.exp_logs.loc[self.exp_logs['exp_id']==self.test_exp_id,:]['trainData_id'].values[0])
		valData_id = int(self.exp_logs.loc[self.exp_logs['exp_id']==self.test_exp_id,'valData_id'].values[0])
		testData_id = int(exp['testData_id'])
		X_train,Y_train,X_val,Y_val,X_test,Y_test = self.build_dataset(trainData_id,valData_id,testData_id)
		model = self.get_model()
		output_test = self.test_model(X_test,Y_test,model)
		output_val = self.test_model(X_val,Y_val,model)
		if (self.tester_mode==0) & (self.model_type=='G'):
			# get discriminator
			df_model_addr = pd.read_csv(os.path.join(CWDIR,'./../../../experiments/menu_model_addr.csv'))
			D_path = os.path.join(CWDIR,'../../../',df_model_addr.loc[df_model_addr['exp_id']==self.test_exp_id,'D_path'].values[-1])
			D = load_model(D_path)
			#predict with D
			log_fakeData = pd.concat([pd.DataFrame(D.predict(self.data_mix['generated_samples']),columns=['predict_true','predict_fake','predict_false']),\
					pd.DataFrame([self.GAN.pdata.decode_mat2seq(x)  for x in self.data_mix['generated_samples']],columns=['generated_samples'])],\
					axis=1)
			log_fakeData.to_csv(os.path.join(CWDIR,'./logs/fakeData/fakeData_exp{}.csv'.format(self.test_exp_id)),index=False)		

		
		#output test
		output_test['results'].to_csv('./logs/predictions/T{}_pred{}_{}.csv'.format(self.model_type,'Test',self.test_exp_id),index=False)
		output_val['results'].to_csv('./logs/predictions/T{}_pred{}_{}.csv'.format(self.model_type,'Val',self.test_exp_id),index=False)

		exp['start_time'] = str(start_time.replace(microsecond=0))
		exp['end_time'] = str(datetime.datetime.now().replace(microsecond=0))
		exp['AUC-test'] = output_test['AUC']
		exp['rho-test'] = output_test['rho']
		exp['pValue-test'] = output_test['pValue']
		exp['acc-test'] = output_test['acc']
		exp['AUC-val'] = output_val['AUC']
		exp['rho-val'] = output_val['rho']
		exp['pValue-val'] = output_val['pValue']
		exp['acc-val'] = output_val['acc']
		exp['n_tested']+=1

		return exp 



if __name__ == '__main__':
	try:
		CWDIR = os.path.abspath(os.path.dirname(__file__))
	except:
		CWDIR = os.getcwd()
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option("-f", "--file_addr", dest="f_addr", type='string', help="The input experiment log file.", default="./exp_logs_TB0.xlsx")
	(options, args) = parser.parse_args(sys.argv)
	f_addr = options.f_addr
#	f_addr = './exp_logs_TG1.xlsx'
	exp_addr = os.path.join(CWDIR,f_addr)
	df = pd.read_excel(exp_addr)
	tGAN = test_enhanceGAN()
	for i in range(df.shape[0]):
		if df.iloc[i]['n_tested'] == 0:
			exp = df.iloc[i]
			print(exp)
			if exp['tester_mode'] ==0:
				exp_ = exp.copy()
				exp_['tester_mode'] = 1
				try:
					exp = tGAN.start(exp_)
					exp['tester_mode'] = 0
					print('Skipped training by taking trained tester.')
				except Exception as e:
					print('Trying to replace tester_mode to 0 but failed, retrain the model.')
					try:
						exp = tGAN.start(exp)
					except Exception as e:
						print(e)
			else:
				try:
					exp = tGAN.start(exp)
				except Exception as e:
					print(e)
				
			df.loc[i,:] = exp
#			df = df.sortlevel(axis=1)
			df.to_excel(exp_addr,index=False)

