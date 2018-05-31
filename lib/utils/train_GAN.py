import pickle
import numpy as np
import pandas as pd


import re
import datetime

import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# these environment parameters shold be imported before keras and tensorfow are imported
#os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5'
#import tensorflow as tf
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
 
from keras import backend as K
#from keras.utils import np_utils
from keras.models import Model, Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import binary_crossentropy,mean_squared_error,kullback_leibler_divergence, categorical_crossentropy

import tensorflow as tf
from keras import metrics

num_cores = 16
GPU = True
if GPU:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 1
    num_GPU = 0
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
		log_device_placement=False,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
K.set_session(tf.Session(config=config))


#from utils import import_local_package
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

import_local_package(os.path.join(CWDIR,'./utils.py'),['get_paths','train_model','test_model','get_batch_idx','save_df'])
import_local_package(os.path.join(CWDIR,'./../../data/lib/prepare_data.py'),['prepare_data'])

#with open(os.path.join(CWDIR,'./../../lib/.model_id'),'r') as f:
#	model_id =f.read().strip()

import_local_package(os.path.join(CWDIR,'./../../experiments/models/{}.py'.format('GAN_1')),['create_G','create_D','create_T'])


class train_GAN(object):
	# class for GAN training, which defines 
	# 1)preparing data 2) initializing 3)GAN training process 4)monitoring 
	# 5)metrics logging 6)categorical weight control 7) balance control
	# Arguments specify the below
	def __init__(self,\
				X_encode = 'onehot',\
				Y_encode = 'onesided',\
				exp_id = '0',\
				D_class_weight={'true':1.2,'fake':1,'false':1.2},\
				G_class_weight={'true':1,'fake':1,'false':1},\
				save_model =True,\
				noise_ratio = 1,\
				is_generating_stochastic = False):
		self.noise_ratio = noise_ratio  #the ratio of noise label size and positive label size, default as 1.
		self.n_trainG = 1 # parameter in Balance Control, number of updates for generator
		self.n_trainD = 1 # parameter in Balance Control, number of updates for discriminator
		self.save_model = save_model #"True", GAN training will save optimal models and logs
		self.is_test_batch = False #parameter used in deciding which epoch to train a evaluating tester
		self.early_quit = False #when set True, the training breaks at epoch level
		self.best_accs_true_avg = 0 #track of best accuracy for true class 
		self.best_accs_false_avg = 0 #track of best accuracy for false class
		self.records = {'score':100,'AUC':0,} #initialize track of AUC records
		self.Y_encode = Y_encode #how to encode Y. when "integer" means maintain original {0,1} label, "onesided" means use {0.1,1}
		self.paths = get_paths(exp_id) # initialize saving paths for experiment id
		self.D_class_weight0 = {0:D_class_weight['true'], 1:D_class_weight['fake'], 2:D_class_weight['false']}  #parameter used in categorical weight control
		self.D_class_weight = dict(self.D_class_weight0)
		self.G_class_weight0 = {0:G_class_weight['true'], 1:G_class_weight['fake'], 2:G_class_weight['false']} #parameter used in categorical weight control
		self.G_class_weight = dict(self.G_class_weight0) 
		self.is_generating_stochastic = is_generating_stochastic #"True", draw generator output sigmoid layer as a distribution; "False" use simply a threshold.
		self.T = None
		self.G = None
		self.D = None
		self.accs_true_ls = []
		self.accs_false_ls = []
		self.Dloss_ls=[] #track of gradient of discriminator
		self.Gloss_ls=[] #track of loss of generator
		self.Dgrad_ls=[] #track of gradient of discriminator
		self.Ggrad_ls=[] #track of gradient of generator
		self.best_equilibrium_avg = 1 #initialize best moving average equilibrium
		self.equilibrium_ls = [1] #track of equilibrium measurement (acc_fake2true - acc_fake)
		self.pdata = prepare_data(X_encode = X_encode,Y_encode = Y_encode) #tool for preparing GAN training data
		self.X_encode = X_encode #how to encode X, default to encode as onehot, deprecated.
		init_op = tf.initialize_all_variables()
		sess = tf.Session()
		sess.run(init_op)
		K.set_session(sess)
#		self.embedding_addr = embedding_addr
#		self.embedding_weights = load_embedding()


	def init_log(self,G,D,data):
		# initialize log of metrics by first calculating 
		# 1)by-class accuracies of GAN 2)confident level divergence (CLD, which measures decisive discriminator is by how divergent its output looks like)
		output={}
		#get prediction
		preds = {}
		generated_samples = G.predict(data['X_noise'])
		preds.update({'true':D.predict(data['X_true']),
						'false':D.predict(data['X_false']),
						'fake':D.predict(generated_samples)})
		"""
		def probstd_transform(generated_samples):
			norm_x = generated_samples/generated_samples.sum(axis=len(generated_samples.shape)-1,keepdims=True)
			choice_stds = norm_x.std(axis=len(generated_samples.shape)-1)
			return choice_stds.mean()

		output.update({'prob_std':probstd_transform(generated_samples)})
		"""
		#get predicted class
		pred_classes = {}
		pred_classes.update({'true':np.array([np.argmax(x) for x in preds['true']])})
		pred_classes.update({'fake':np.array([np.argmax(x) for x in preds['fake']])})
		pred_classes.update({'false':np.array([np.argmax(x) for x in preds['false']])})
		pred_is_correct = {}
		pred_is_correct.update({'true':np.array([x==0 for x in pred_classes['true']])})
		pred_is_correct.update({'fake':np.array([x==1 for x in pred_classes['fake']])})
		pred_is_correct.update({'false':np.array([x==2 for x in pred_classes['false']])})
		pred_is_correct.update({'fake2true':np.array([x==0 for x in pred_classes['fake']])})
		pred_is_correct.update({'fake2false':np.array([x==2 for x in pred_classes['fake']])})
		#get accuracy
		output.update({'accs_'+y:sum(pred_is_correct[y])/len(pred_is_correct[y]) for y in ['true','fake','false','fake2true','fake2false']})
		#get KL divergence of prediction
		"""
		def ConfidenceLevel_divergence(p,q,cutoff=0.1):
			judge_mat = q>0.5
			div_true =(1-p[judge_mat])
			div_fakefalse = p[~judge_mat]
			x = np.concatenate([div_true,div_fakefalse])
			#return np.log(x[x>1])
			if (x > cutoff).sum() >0:
				output = -1*(x[x>cutoff]*np.log(x[x>cutoff]))
			else:
				output = np.array([0])

			return output.sum()

		output.update({'CLD_'+y:ConfidenceLevel_divergence(p=preds[y],q=data['Y_'+y],cutoff=0.1) for y in ['true','fake','false']})
		"""
		return output
	"""
	def scorer(self,log_dict):
		# score defined by averaging confidence level divergence
		return np.array([log_dict[y] for y in ['CLD_true','CLD_fake','CLD_false']]).mean()
	"""
	
	def get_tester_output(self,G,D,log_dict):
		#calculate AUC, rho of pearson coefficient, pValue of rho of pearson coefficient, accuracy on validation dataset
		data_mix = self.mix_data(G,self.train_data)

		log_dict['generated_samples'] = self.pdata.decode_mat2seq(data_mix['generated_samples'][0])
		T = train_model(
			model=self.T,\
			X_train=data_mix['X_train'],\
			Y_train=data_mix['Y_train'],\
			n_epoch =self.n_epoch_T,\
			monitor='val_acc',\
			min_delta=0.001,\
			patience=20,\
			verbose=2,\
			mode='auto',\
			shuffle = True,\
			validation_split=0.2,\
			log_path = self.paths['log_T_path'],\
			model_path = self.paths['T_path'])

		output_test = test_model(X_test = self.X_val,Y_test=self.Y_val,model=T,cutoff=0.5)
		log_dict['AUC'] = output_test['AUC']
		log_dict['rho'] = output_test['rho']
		log_dict['pValue'] = output_test['pValue']
		log_dict['acc'] = output_test['acc']
		self.is_test_batch = False

		return log_dict


	def get_batch_log(self,G,D,GAN_data_i):
		#get logs of training metrics
		log_dict = self.init_log(G,D,GAN_data_i['data'])
		log_dict['AUC'] = 0
		log_dict['rho'] = 0
		log_dict['pValue'] = 1
		log_dict['acc'] = 0
		log_dict['generated_samples'] = ''
		log_dict['i_batch'] = GAN_data_i['i']
		log_dict['i_epoch'] = GAN_data_i['_e']
		log_dict['G_loss'] = self.Gloss_ls[-1]
		log_dict['G_gradient'] = self.Ggrad_ls[-1]
		log_dict['D_loss'] = self.Dloss_ls[-1]
		log_dict['D_gradient'] = self.Dgrad_ls[-1]
		log_dict['D_class_weights'] = self.D_class_weight
		log_dict['G_class_weights'] = self.G_class_weight
		log_dict['datetime'] = datetime.datetime.now()
		#log_dict['score'] = self.scorer(log_dict)

		#replace equilibrium and save
		if log_dict['accs_fake2false'] ==0:
			eql_rate = abs(log_dict['accs_fake2true']-log_dict['accs_fake'])
		else:
			eql_rate = 1
		log_dict['equilibrium_avg'] = (np.sum(self.equilibrium_ls)+eql_rate)/(self.n_batch+1)
		log_dict['accs_false_avg'] = (np.sum(self.accs_false_ls)+log_dict['accs_false'])/(self.n_batch+1)
		log_dict['accs_true_avg'] = (np.sum(self.accs_true_ls)+log_dict['accs_true'])/(self.n_batch+1)
		print('best_equilibrium_avg',self.best_equilibrium_avg)
		print('now_equilibrium_avg',log_dict['equilibrium_avg'])
		print(self.n_trainD)
		print(self.n_trainG)
		print('best_accs_true_avg:',self.best_accs_true_avg)
		print('best_accs_false_avg:',self.best_accs_false_avg)
		print('accs_false_avg:',log_dict['accs_false_avg'])
		print('accs_true_avg:',log_dict['accs_true_avg'])

		#replace AUC and save
		if self.T !=None:
			if (self.is_test_batch) \
				& ((log_dict['accs_true']>=0.9)&(log_dict['accs_false']>0.5)&(log_dict['accs_fake2false']<0.2)\
					|(GAN_data_i['i']==GAN_data_i['n_batch']-1)):
					log_dict = self.get_tester_output(G,D,log_dict)
			if (log_dict['AUC'] > self.records['AUC'])\
					&(self.save_model==True):
				print()
				print('***********************************************************')
				print('Get better tester validation AUC {}, G and D replaced.'.format(log_dict['AUC']))
				print('***********************************************************')
				print()
				data_mix = self.mix_data(G,self.train_data)
				G.save(self.paths['G_path'])
				D.save(self.paths['D_path'])
				fakeData_log = pd.concat([pd.DataFrame(D.predict(data_mix['generated_samples']),columns=['predict_true','predict_fake','predict_false']),\
						pd.DataFrame([self.pdata.decode_mat2seq(x)  for x in data_mix['generated_samples']],columns=['generated_samples'])],\
						axis=1)
				fakeData_log.to_csv(self.paths['fakeData_path'],index=False)
				self.G, self.D =G,D
				self.records = log_dict
				self.best_equilibrium_avg = log_dict['equilibrium_avg']
								
		elif self.T ==None:
			if ((log_dict['equilibrium_avg'] < self.best_equilibrium_avg) \
					& ((self.best_accs_true_avg-log_dict['accs_true_avg']<0.1)&(self.best_accs_false_avg-log_dict['accs_false_avg']<0.2)))\
						& (self.save_model==True):
				print()
				print('***********************************************************')
				print('Get better average equilibrium {}, G and D replaced.'.format(log_dict['equilibrium_avg']))
				print('***********************************************************')
				print()
	#			data_mix = self.mix_data(G,self.train_data)
				G.save(self.paths['G_path'])
				D.save(self.paths['D_path'])
	#			fakeData_log = pd.concat([pd.DataFrame(D.predict(data_mix['generated_samples']),columns=['predict_true','predict_fake','predict_false']),\
	#					pd.DataFrame([self.pdata.decode_mat2seq(x)  for x in data_mix['generated_samples']],columns=['generated_samples'])],\
	#					axis=1)
	#			fakeData_log.to_csv(self.paths['fakeData_path'],index=False)
				self.G, self.D =G,D
				self.records = log_dict
				self.best_equilibrium_avg = log_dict['equilibrium_avg']
		self.equilibrium_ls.append(eql_rate)
		self.equilibrium_ls = self.equilibrium_ls[-self.n_batch:]

		if (log_dict['accs_false_avg'] > self.best_accs_false_avg)&(log_dict['accs_true_avg'] > self.best_accs_true_avg):
			self.best_accs_false_avg = log_dict['accs_false_avg']
			self.best_accs_true_avg = log_dict['accs_true_avg']

		self.accs_true_ls.append(log_dict['accs_true'])
		self.accs_true_ls = self.accs_true_ls[-self.n_batch:]
		self.accs_false_ls.append(log_dict['accs_false'])
		self.accs_false_ls = self.accs_false_ls[-self.n_batch:]		

		print('Nash Equiblirium measure (patient_batch:{}):{}'.format(self.n_batch,self.equilibrium_ls))
			
		return log_dict

	def build_gradients_getter(self,model,weights):
		#define a keras layer that extract gradients from a given layer
		#model: model that contrains the layer
		#weights:tensor layers
		gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
		input_tensors = [model.inputs[0], # input data
						model.sample_weights[0], # how much to weight each sample by
						model.targets[0], # labels
						K.learning_phase(), # train or test mode
		]
		get_gradients = K.function(inputs=input_tensors, outputs=gradients)
		
		return get_gradients

	def monitor(self,log_dict):
		# print metrics logged in training
		print('\n')
		print('\n')
		print('For',log_dict['i_epoch'],'epoch,',log_dict['i_batch'],'batch: ')
		
		print('    D_class_weights:',self.D_class_weight)
		print('    G_class_weights:',self.G_class_weight)
		"""
		print('    CLD_true: {}, CLD_fake: {}, CLD_false: {}'\
					.format(
						log_dict['CLD_true'],\
						log_dict['CLD_fake'],\
						log_dict['CLD_false']\
					))

		"""
		print('    D_loss: {}, D_gradient: {}'.format(log_dict['D_loss'],log_dict['D_gradient']))
		print('    G_loss:',log_dict['G_loss'], 'G_gradient:',log_dict['G_gradient'])
		print('    Accuracy by category: ')
		print('        accs_true: {},accs_fake: {} ,acc_false: {}'\
							.format(
								log_dict['accs_true'],\
								log_dict['accs_fake'],\
								log_dict['accs_false']\
							))
		print('        accs_fake2true: {} ,accs_fake2false: {}'\
							.format(
								log_dict['accs_fake2true'],\
								log_dict['accs_fake2false']))
		try:
			print('        Tester validation AUC: {}'.format(log_dict['AUC']))
			print('        generated samples:{}'.format(log_dict['generated_samples']))
		except:
			print('No test')
			

		print('\n')

		return

	def norm_add(self,num,pos):
		# function used to normalize categorical_weight_control
		_dict = self.D_class_weight
		_dict[pos]+=num
		_sum = sum(list(_dict.values()))
		return {i:_dict[i]*sum(list(self.D_class_weight0))/_sum for i in range(3)}


	def categorical_weight_control(self,log_dict):
		#policies that pay different attention to loss from different class of Y 
		#policies based on class-wise accuracies

		#check real data prediction
		if log_dict['accs_true'] < 0.9:
			self.D_class_weight = self.norm_add(10,0)

		if log_dict['accs_true'] >= 0.9:
			self.D_class_weight[0] = dict(self.D_class_weight0)[0]
		#check fake data prediction
		if (log_dict['accs_true']>=0.7) & (log_dict['accs_fake'] <0.1):
			self.D_class_weight = self.norm_add(4,1)
		if (log_dict['accs_fake2false'] >0):
			self.D_class_weight = self.norm_add(4,2)
		else:
			self.G_class_weight[2] = dict(self.G_class_weight0)[2]

		if (log_dict['accs_true']<0.7) | (log_dict['accs_fake'] >0.8):
			self.D_class_weight[1] = dict(self.D_class_weight0)[1]
			self.G_class_weight[0] = dict(self.G_class_weight0)[0]
		#check opposite data prediction
		if (log_dict['accs_true']>=0.9) & (log_dict['accs_false'] < 0.6) :
			self.D_class_weight = self.norm_add(4,2)
		if (log_dict['accs_true']<0.9) | (log_dict['accs_false'] >= 0.6):
			self.D_class_weight[2] = dict(self.D_class_weight0)[2]

		return

	def balance_control(self,log_dict):
		#policies to balance the training of generator and discriminator to make them match
		#polciies based on gradients and losses and class-wise accuracies

		#when D is not good enough to describe real, G stops
		if (log_dict['accs_false_avg']<0.6)|(log_dict['accs_true_avg']<0.8):
			self.n_trainD = 1
			self.n_trainG = 0
		#when D too fast, G accelerates
		elif (log_dict['accs_fake2false']>0)\
				|((log_dict['accs_fake']==1)&(np.std(self.Gloss_ls)<0.05)&(np.mean(self.Ggrad_ls)<0.001)):
			self.n_trainD = 0
			self.n_trainG = 3
		#when G too fast, G stops, leaving D time to study on it
		elif (log_dict['accs_fake2true']==1)&(np.std(self.Gloss_ls)<0.05)&(np.mean(self.Ggrad_ls)<0.001):
			self.n_trainD = 1
			self.n_trainG = 0
		else:
			self.n_trainD = 1
			self.n_trainG = 1
		return 




	def early_stopping(self,log_dict):
		#early stopping policies
		#polciies based on gradients and losses and class-wise accuracies
		 
		if (log_dict['accs_fake']==1)&(np.std(self.Gloss_ls)<0.002)&(np.mean(self.Ggrad_ls)<0.00001):
			print('Early stopping mode 1...')
			self.early_quit = True
		if (log_dict['accs_fake2true']==1)&(np.std(self.Gloss_ls)<0.002)&(np.mean(self.Ggrad_ls)<0.00001):
			print('Early stopping mode 2...')
			self.early_quit = True
		if (log_dict['equilibrium_avg']<0.05):
			self.early_quit = True
			print('Early stopping mode 3...')
#		if np.std(self.Dloss_ls) > 10:
#			self.early_quit = True
#		if np.mean(self.Dloss_ls) > 100:
#			self.early_quit = True
		return




	def init_GAN_trainingData(self,X_train,Y_train,G_input_dim=100):
		# create noise for generator input
		# transfer binary label to multi-class label
		# G_input_dim is the dimension of generator input, the higher the more abundant patterns of fake expected in generator output
		if self.Y_encode == 'onesided':
			Y_encoder = (1,.1)
		else:
			Y_encoder = (1.,0.)
		X_true = X_train[Y_train==Y_encoder[0]]
		X_false = X_train[Y_train==Y_encoder[1]]
#		X_false = X_train[Y_train==0]
		npr = np.random.RandomState(123)
		noise_size = int(X_train.shape[0]/2)
		X_noise = np.array([npr.uniform(-1,1,G_input_dim) for i in range(noise_size*self.noise_ratio)])
		train_data = {'X_true':X_true,'X_false':X_false,'X_noise':X_noise}
		train_data.update({'Y_true':np.array([np.array([Y_encoder[0],Y_encoder[1],Y_encoder[1]])]*train_data['X_true'].shape[0])})
		train_data.update({'Y_fake':np.array([np.array([Y_encoder[1],Y_encoder[0],Y_encoder[1]])]*train_data['X_noise'].shape[0]*self.noise_ratio)})
		train_data.update({'Y_false':np.array([np.array([Y_encoder[1],Y_encoder[1],Y_encoder[0]])]*train_data['X_false'].shape[0])})
		return train_data

	def mix_data(self,G,train_data):
		# combine different class of X to one big X
		gen_data = self.pdata.generate_data(G=G,noises=train_data['X_noise'],X_encode=self.X_encode,
											is_generating_stochastic=self.is_generating_stochastic)
		###sample X_fake
		train_data.update({'X_generated':gen_data})
		del gen_data
		X_train_mix = np.concatenate([train_data['X_true'],train_data['X_generated'],train_data['X_false']])
		###encode onehot to integer
#		X_train_mix = np.argmax(X_train_CNN,axis=len(X_train_CNN.shape)-1)
		###get Y
		Y_train_mix = np.array(len(train_data['X_true'])*[1]+len(train_data['X_noise'])*[1]+len(train_data['Y_false'])*[0])
		return {'X_train':X_train_mix,'Y_train':Y_train_mix,'generated_samples':train_data['X_generated']}

	def init_GD_models(self,G,D,X_train,Y_train,n_epoch_init_G=10,n_epoch_init_D=10):
		#pretrain generator and discriminator
		#n_epoch_init_G: training maximial epoch number for generator
		#n_epoch_init_D: training maximal epoch number for discriminator

		train_data = self.init_GAN_trainingData(X_train,Y_train)

		min_num = min(train_data['X_noise'].shape[0],train_data['X_true'].shape[0])
		X_G_init = np.concatenate([train_data['X_noise'][:min_num]])
		Y_G_init = np.concatenate([train_data['X_true'][:min_num]])

		X_D_init = np.concatenate([train_data['X_true'],train_data['X_false']])
		Y_D_init = np.concatenate([train_data['Y_true'],train_data['Y_false']])
		del train_data
		Y_D_init[Y_D_init>0.5]=1
		Y_D_init[Y_D_init<0.5]=0
		D = train_model(D,\
					X_D_init,\
					Y_D_init,\
					n_epoch =  n_epoch_init_D,\
					monitor='loss',\
					min_delta=0.0005,\
					patience=15,\
					verbose=2,\
					mode='auto',\
					shuffle = True,\
					validation_split=0.1,\
					log_path=self.paths['log_init_D_path'],\
					model_path = self.paths['init_D_path'])

		G = train_model(G,\
					X_G_init,\
					Y_G_init,\
					n_epoch = n_epoch_init_G,\
					monitor='loss',\
					min_delta=0.0005,\
					patience=15,\
					verbose=2,\
					mode='auto',\
					shuffle = True,\
					validation_split=0.1,\
					log_path=self.paths['log_init_G_path'],\
					model_path = self.paths['init_G_path'])

		return G,D
	def get_GAN_batches(self,X_train,Y_train,batch_size=256,val_prop=0.1):
		#divide GAN shaped data to batches by batch_size
		# batch_size: how many true class and false class data in one batch (noticed it doesn't consider fake samples size)
		# val_prop:: how much proportion of real data is used for in-training validation
		

		#get the data
		train_data = self.init_GAN_trainingData(X_train=X_train,Y_train=Y_train)
		sizes_dict = {x:len(train_data[x]) for x in ['X_true', 'X_false', 'X_noise']}
		n_batches_dict = {x:int(batch_size*sizes_dict[x]/sizes_dict['X_true']) for x in ['X_true', 'X_false', 'X_noise']}
		indexes_dict = {x:list(range(sizes_dict[x])) for x in ['X_true', 'X_false', 'X_noise']}
		n_batch = int(X_train.shape[0]/batch_size)
		n_batch_val = int(n_batch*val_prop) # n_batch_val can be 0, the validation step has to be able to deal with that
		b_idx_dict = {x:get_batch_idx(indexes_dict[x],n_batch) for x in ['X_true', 'X_false', 'X_noise']}
		train_batches = []
		for i in range(0,n_batch-n_batch_val):
			indexes_i = {x:b_idx_dict[x][i] for x in ['X_true', 'X_noise', 'X_false']}
			train_data_i = {}
			for x in ['X_true', 'X_noise', 'X_false']:
				idx = indexes_i[x]
				train_data_i.update({x:train_data[x][idx]})
				if x !='X_noise':
					y = x.replace('X','Y')
				else:
					y = 'Y_fake'
				train_data_i.update({y:train_data[y][idx]})
			train_batches.append(train_data_i)

		val_batch = []
		for i in range(n_batch-n_batch_val,n_batch):
			indexes_i = {x:b_idx_dict[x][i] for x in ['X_true', 'X_noise', 'X_false']}
			val_data_i = {}
			for x in ['X_true', 'X_noise', 'X_false']:
				idx = indexes_i[x]
				val_data_i.update({x:train_data[x][idx]})
				if x !='X_noise':
					y = x.replace('X','Y')
				else:
					y = 'Y_fake'
				val_data_i.update({y:train_data[y][idx]})
			val_batch.append([val_data_i])		
		val_batch = sum(val_batch,[])
		return {'train_batches':train_batches,'val_batch':val_batch}


	def train_GAN_one_batch(self,G,D,dcgan,GAN_data_i):		
		# define the update of GAN per batch

		data = GAN_data_i['data']
		data.update({'X_fake':G.predict(data['X_noise'])})
		#train the discriminator first
		if self.n_trainG == 0:
			X = np.concatenate(( data['X_true'], data['X_false']), axis = 0)
			Y = np.concatenate(( data['Y_true'], data['Y_false']))
		else:			
			X = np.concatenate(( data['X_true'], data['X_fake'], data['X_false']), axis = 0)
			Y = np.concatenate(( data['Y_true'], data['Y_fake'], data['Y_false']))
		if (self.n_trainD != 0):
			D_loss = D.train_on_batch(X,Y,class_weight=self.D_class_weight)
		else:
			D_loss = 0
		D.trainable = False  #fix discriminator first

#				D_loss = D.train_on_batch(X,
#											np.concatenate(( data['Y_true'], GdataAN_data_i['Y_fake'], data['Y_false'])),
#											class_weight=self.D_class_weight)
		i=0
		while i <self.n_trainG:
			GAN_loss = dcgan.train_on_batch(data['X_noise'],
												np.array([data['Y_true'][0]]*data['X_noise'].shape[0]),#to fool the discriminator, we put the label to be true rather than fake
												class_weight=self.G_class_weight) #recorded dcganLoss
			i+=1 
		D.trainable = True
		if (self.n_trainG==0):
			GAN_loss = 0
		g_change = np.mean([abs(x).mean() for x in self.get_gradients_G([GAN_data_i['data']['X_noise'],[1],GAN_data_i['data']['Y_fake'],0])])
		d_change = np.mean([abs(x).mean() for x in self.get_gradients_D([X,[1],Y,0])])
		self.Dloss_ls.append(D_loss)
		self.Gloss_ls.append(GAN_loss)
		self.Dgrad_ls.append(d_change)
		self.Ggrad_ls.append(g_change)
		self.Dloss_ls= self.Dloss_ls[-self.patient*self.n_batch:]
		self.Gloss_ls=self.Gloss_ls[-self.patient*self.n_batch:]
		self.Dgrad_ls=self.Dgrad_ls[-self.patient*self.n_batch:]
		self.Ggrad_ls=self.Ggrad_ls[-self.patient*self.n_batch:]

		return G,D,dcgan


	def train_GAN(self,X_train,Y_train,G,D,X_val=None,Y_val=None,T=None,\
					batch_size = 256,\
					val_prop=0.1,\
					n_epoch=200,\
					patient=5,\
					n_epoch_T=50,\
					is_categorical_weight_control=False,\
					is_monitored=True,
					is_early_stopping=True):
		self.is_early_stopping = is_early_stopping
		# define the training process
		self.patient = patient #parameter used for tracking GAN training, e.g. how many epoch to track for gradients,losses,equilibriums
		self.X_val = X_val 
		self.Y_val = Y_val 
		self.n_epoch_T=n_epoch_T #number of epochs for tester training
		self.T = T
		#get the model
		DCGAN_input = Input(shape=[x for x in G.get_input_shape_at(0) if x!=None]+[])
		DCGAN_output = D(G(DCGAN_input))
		dcgan = Model(input=DCGAN_input, output=DCGAN_output)
		dcgan.compile(loss=G.loss, optimizer=G.optimizer)

		weights_G = [weight for weight in dcgan.layers[1].trainable_weights if dcgan.layers[1].get_layer(re.search('.*(?=\/)',weight.name).group()).trainable] # filter down weights tensors to only ones which are trainable
		weights_D = [weight for weight in D.trainable_weights if D.get_layer(re.search('.*(?=\/)',weight.name).group()).trainable] # filter down weights tensors to only ones which are trainable

		self.get_gradients_G = self.build_gradients_getter(dcgan,weights_G)
		self.get_gradients_D = self.build_gradients_getter(D,weights_D)
		
		#start training
		_e = 0
		while _e < n_epoch:
			if _e == 1:
				self.G, self.D = None,None # to prevent the Nash Equilibrium achieved at the 0th epoch...
				self.best_equilibrium_avg = 1 #reinitialize best moving average equilibrium
			self.train_data = self.init_GAN_trainingData(X_train=X_train,Y_train=Y_train)
			data_batches=self.get_GAN_batches(X_train,Y_train,batch_size=batch_size,val_prop=val_prop)
			self.n_batch = len(data_batches['train_batches'])
			i = 0
			if (_e % 5==0):
				self.is_test_batch = True
			else:
				self.is_test_batch = False
			while i < self.n_batch:
				#train models
				GAN_data_i = {'data':data_batches['train_batches'][i]}
				GAN_data_i.update({'_e':_e})
				GAN_data_i.update({'i':i})
				GAN_data_i.update({'n_batch':self.n_batch})
				G,D,dcgan= self.train_GAN_one_batch(G,D,dcgan,GAN_data_i)
				
				log_dict = self.get_batch_log(G,D,GAN_data_i)
				if self.save_model==True:																
					save_df(self.paths['log_GAN_train_path'],pd.DataFrame([log_dict]))
				if is_categorical_weight_control:
					self.categorical_weight_control(log_dict)
				if is_monitored:
					self.monitor(log_dict)
				is_balance_control = True
				if is_balance_control == True:
					self.balance_control(log_dict)


				i+=1
			log_dict_test = {}
			if len(data_batches['val_batch']) > 0:
				GAN_data_i = {'data':data_batches['val_batch'][0]}
				GAN_data_i.update({'_e':_e})
				GAN_data_i.update({'i':i})
				GAN_data_i.update({'n_batch':self.n_batch+1})
				_,_,_ = self.train_GAN_one_batch(G,D,dcgan,GAN_data_i)
				log_dict_test = self.get_batch_log(G,D,GAN_data_i)			
			else:
				for k,v in log_dict.items():
					log_dict_test[k] = None
			if self.save_model ==True:
				save_df(self.paths['log_GAN_val_path'],pd.DataFrame([log_dict_test]))
			del log_dict_test		
			if self.is_early_stopping == True:
				self.early_stopping(log_dict)
			if self.early_quit ==True:
				_e = n_epoch
			else:
				_e +=1
		if self.G == None:
			print()
			print('***********************************************************')
			print('GAN has never seen equilibrium during training, quit with final G, D')
			print('***********************************************************')
			print()
			G.save(self.paths['G_path'])
			D.save(self.paths['D_path'])
			self.G, self.D =G,D			

		return {'G':G,'D':D,'records':self.records}

if __name__ == '__main__':

	adam0=Adam(lr=0.005, beta_1=0.9 ,decay=0.001)
	adam1=Adam(lr=0.0005, beta_1=0.9 ,decay=0.001)
	sgd0 = SGD(lr=0.001, decay=0.01, momentum=0.9, nesterov=True)
	sgd1 = SGD(lr=0.000015, decay=0.01, momentum=0.9, nesterov=True)
	rmsprop0 = RMSprop(lr=0.0004, rho=0.9, epsilon=1e-08, decay=0.01)
	rmsprop1 = RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.01)

	del prepare_data
	import_local_package(os.path.join(CWDIR,'./../../data/lib/prepare_data.py'),['prepare_data'])
	pdata=prepare_data(X_encode = 'onehot',Y_encode = 'integer')
	_,_,X_val,Y_val,X_test,Y_test = pdata.get_data(0,0,0)

	pdata=prepare_data(X_encode = 'onehot',Y_encode = 'onesided')
	X_train,Y_train,_,_,_,_ = pdata.get_data(2,2,2)
	length = X_train.shape[1]

	GAN = train_GAN(
				X_encode = 'onehot',\
				Y_encode = 'onesided',\
				exp_id = '0',\
				D_class_weight={'true':1.2,'fake':1,'false':1.2},\
				G_class_weight={'true':1,'fake':1,'false':1},\
				noise_ratio=2,\
				save_model=True,\
				is_generating_stochastic = False)
	G = create_G(input_dim =100,output_dim= (length,20))
	G.compile(loss='categorical_crossentropy', optimizer = adam0)
	D = create_D(input_dim =(length,20),output_dim=3 )
	D.compile(loss='categorical_crossentropy', optimizer = rmsprop0)   #this will lead to poor performance of Discriminator at about epoch 39
	G,D = GAN.init_GD_models(G,D,X_train,Y_train,n_epoch_init_G=2,n_epoch_init_D=1)

	G.compile(loss='categorical_crossentropy', optimizer = adam0)
	D.compile(loss='categorical_crossentropy', optimizer = adam1)   #this will lead to poor performance of Discriminator at about epoch 39
	T = create_T(input_dim =(length,20),output_dim=1 )
	T.compile(loss='binary_crossentropy', optimizer = rmsprop0, metrics=['accuracy'])


	output =GAN.train_GAN(
			X_train=X_train,
			Y_train=Y_train,
			G=G,
			D=D,\
			X_val=X_val,\
			Y_val=Y_val,\
			T = T,\
			batch_size = 128,\
			val_prop = 0,\
			n_epoch=10,\
			patient=15,\
			n_epoch_T=2,\
			is_categorical_weight_control=False,
			is_monitored=True,\
			is_early_stopping=True)
	from utils import test_model_D

	train_data = GAN.init_GAN_trainingData(X_train=X_val,Y_train=Y_val)
	X = np.concatenate([train_data['X_true'],train_data['X_false']])
	Y = np.concatenate([train_data['Y_true'],train_data['Y_false']])
	output = test_model_D(X_test = X,Y_test=Y,model=D,cutoff=0.5)

