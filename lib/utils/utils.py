import numpy as np
import pandas as pd

import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	

def import_local_package(addr_pkg,function_list=[]):
	#import local package by address
	#it has to be imported directly in the file that contains functions required the package, i.e. it cannot be imported by from .../utils import import_local_package
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

def get_batch_idx(data,n):
	#divide a sequence data in to n equal parts
	def chunkIt(seq, n):
		avg = len(seq) / float(n)
		out = []
		last = 0.0

		while last < len(seq):
			out.append(seq[int(last):int(last + avg)])
			last += avg
		return out
	rnp = np.random.RandomState(123)
	seq = list(rnp.choice(data,len(data),replace=False))
	return chunkIt(seq,n)



def save_df(addr,df):
	#save a dataframe to an address
	#if file existed, append on it
	if os.path.isfile(addr):
		df.to_csv(addr,header=False,mode='a',index=False,sep=',')
	else:
		pd.DataFrame(columns=df.columns).to_csv(addr,header=True,index=False,sep=',')
		df.to_csv(addr,header=False,mode='a',index=False,sep=',')
	return

def get_paths(exp_id):
	# initialize paths involved in GAN training
	log_dir_path = os.path.join(CWDIR,'./../../logs/test_history/')
	model_dir_path = os.path.join(CWDIR,'./../../logs/trained_models/')
	fakeData_dir_path = os.path.join(CWDIR,'./../../logs/fakeData/')
	output = {
			#models:
			'G_path':os.path.join(model_dir_path,'GAN','GAN_G_exp'+str(exp_id)+'.h5'),
			'D_path':os.path.join(model_dir_path,'GAN','GAN_D_exp'+str(exp_id)+'.h5'),

			'T_path':os.path.join(model_dir_path,'tester','GAN_T_exp'+str(exp_id)+'.h5'),

			'init_G_path':os.path.join(model_dir_path,'init','init_G_exp'+str(exp_id)+'.h5'),
			'init_D_path':os.path.join(model_dir_path,'init','init_D_exp'+str(exp_id)+'.h5'),

			#logs
			'log_GAN_train_path':os.path.join(log_dir_path,'GAN','log_GAN_val_exp'+str(exp_id)+'.csv'),
			'log_GAN_val_path':os.path.join(log_dir_path,'GAN','log_GAN_valTrain_exp'+str(exp_id)+'.csv'),
			'log_T_path':os.path.join(log_dir_path,'tester','log_T_exp'+str(exp_id)+'.csv'),

			'log_init_G_path':os.path.join(log_dir_path,'init','log_init_G_exp'+str(exp_id)+'.csv'),
			'log_init_D_path':os.path.join(log_dir_path,'init','log_init_D_exp'+str(exp_id)+'.csv'),

			#fake data samples
			'fakeData_path':os.path.join(fakeData_dir_path,'fakeData_exp'+str(exp_id)+'.csv'),
			}
	for k,v in output.items():
		_dir, _f = os.path.split(v)
		output[k] = os.path.realpath(v)
		os.system('mkdir -p {}'.format(_dir))
	return output



def load_embedding():
	# load embedding weights, deprecated
	from gensim.models import Word2Vec
	model = Word2Vec.load(os.path.join(CWDIR,'./../embedding/peptideEmbedding.bin'))
	embedding_weights = np.zeros((model.wv.syn0.shape[0]+1,model.wv.syn0.shape[1]))
	for i in range(len(model.wv.vocab)):
		embedding_vector = model.wv[model.wv.index2word[i]]
		if embedding_vector is not None:
			embedding_weights[i] = embedding_vector
	return embedding_weights

def train_model(model,\
		X_train,Y_train,\
		n_epoch =  200,\
		monitor='loss',\
		min_delta=0.0005,\
		patience=5,\
		verbose=2,\
		mode='auto',\
		shuffle = True,\
		validation_split=0.2,\
		log_path = '',\
		model_path = ''):
	from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
	#train a model
	#parameter names are the same as keras model.fit

	earlyStopping = EarlyStopping(monitor=monitor,
								min_delta = min_delta,
								patience=patience,
								verbose=verbose,
								mode=mode)
	callbacks = [earlyStopping]
	if log_path != '':
		csv_logger = CSVLogger(log_path,append=True)
		callbacks.append(csv_logger)
	if model_path != '':
		checkpointer = ModelCheckpoint(filepath=model_path, save_best_only=True)
		callbacks.append(checkpointer)

	batch_size = int(np.ceil(len(X_train)/100.0)) # variable batch size depending on number of data points
	mod = model.fit(X_train, Y_train,
					batch_size=batch_size,
					epochs = n_epoch,
					verbose=2,
#						callbacks=[earlyStopping,checkpointer,csv_logger],
					callbacks=callbacks,
					shuffle=shuffle, validation_split=validation_split)
	return mod.model

#from keras.models import  load_model
def load_model(model,path_parameter):
	#load model by weights, to avoid bugs relating to initializing keras computing graph when using keras.models.load_model
	model.load_weights(path_parameter,by_name=True)
	return model


def test_model_D( X_test, Y_test, model, cutoff=0.5):
	#test discriminator, it accepts data with multi-class labeled Y, 
	# ittransfers multiclass label into binary by setting fake and true prediction equivalent
	# it then calculates AUC, rho, pValue and acc
	#cutoff: the threshould between (0,1) used for predicting class
	from sklearn.metrics import roc_auc_score, roc_curve, auc
	from scipy import stats
	Y_predict = np.squeeze(model.predict(X_test))
	idxes_min = np.argmin(Y_predict,1)
	Y_pred = Y_predict.copy()
	for i in range(len(idxes_min)):
		del_p = idxes_min[i]
		Y_pred[i][del_p] = 0
	Y_pred = Y_pred/Y_pred.sum(axis=1,keepdims=True)
	idxes = np.argmax(Y_pred,1)
	Y_pred = np.array([1-Y_pred[i][idxes[i]] if idxes[i] == 2 else Y_pred[i][idxes[i]] for i in range(len(idxes))])
	def get_binary(Y_test):
		idxes = np.argmax(Y_test,1)
		tmp = np.array([1 if i ==0 else i for i in idxes])
		return np.array([0 if i ==2 else i for i in tmp])
	Y_test_binary = get_binary(Y_test)
	Y_predict_class = (Y_pred>cutoff).astype(int)
	results = pd.concat([pd.DataFrame(Y_predict,columns=['predict_true','predict_fake','predict_false']),\
						pd.DataFrame(Y_pred,columns=['Y_predict_binary']),\
						pd.DataFrame(Y_predict_class,columns=['Y_predict_class']),\
						pd.DataFrame(Y_test_binary,columns=['Y_test_binary'])
#						pd.DataFrame([pdata.decode_mat2seq(x) for x in X_test],columns=['X_test'])
						],axis=1)
	acc = (results['Y_predict_class']==results['Y_test_binary']).sum()/results.shape[0]
	#auc
	FPR, TPR, thresholds = roc_curve(Y_test_binary, Y_pred, pos_label=1,drop_intermediate=False)
	AUC = auc(FPR, TPR)
	#rho
	rho, pValue = stats.spearmanr(Y_test_binary, Y_pred)
	return {'results':results,'acc':acc,'AUC':AUC,'rho':rho,'pValue':pValue}


def test_model( X_test, Y_test, model, cutoff=0.5):
	#test binary data
	#calculate AUC, rho, pValue and acc
	#cutoff: the threshould between (0,1) used for predicting class
	from sklearn.metrics import roc_auc_score, roc_curve, auc
	from scipy import stats
	Y_predict = np.squeeze(model.predict(X_test))

	Y_predict_class = (Y_predict>cutoff).astype(int)
	results = pd.concat([pd.DataFrame(Y_predict),pd.DataFrame({'Y_test':Y_test,'Y_predict_class':Y_predict_class})],axis=1)
	acc = (results['Y_predict_class']==results['Y_test']).sum()/results.shape[0]
	#auc
	FPR, TPR, thresholds = roc_curve(Y_test, Y_predict, pos_label=1,drop_intermediate=False)
	AUC = auc(FPR, TPR)
	#rho
	rho, pValue = stats.spearmanr(Y_test, Y_predict)
	return {'results':results,'acc':acc,'AUC':AUC,'rho':rho,'pValue':pValue}


