#the file defines functions of preparing data for GAN
import numpy as np
import pandas as pd
import os
from collections import OrderedDict

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
import_local_package(os.path.join(CWDIR,'./../../lib/utils/multitask_utils.py'),['multi_work'])


class prepare_data(object):
	#to customize data, you only need to change self.pos_dict
    #                   and self.build_dataset
	def __init__(self,X_encode = 'onehot',Y_encode = 'integer'):
		self.scales = 2560 # the bigger this number ,the less threads used for generating data
		self.Y_encode = Y_encode #how to encode Y. when "integer" means maintain original {0,1} label, "onesided" means use {0.1,1}
		self.X_encode = X_encode #how to encode X, default to encode as onehot, deprecated.
		self.pos_dict = OrderedDict([
					   ('0', 0),
					   ('1', 1)
					   ]) # encoding
		

	def encode_seq2vec(self,seq):
		vec = []
		for pos in seq:
			vec.append(self.pos_dict[pos])
		vec = np.array(vec)
		return vec

	def encode_seq2mat(self,seq):
		mat = np.zeros((len(seq),20))
		for i in range(len(seq)):
		  pos = seq[i]
		  mat[i,self.pos_dict[pos]] = 1.
		return mat
	
	def decode_mat2seq(self,mat):
		rev_dict = {}	
		for k,v in self.pos_dict.items():
			rev_dict[v] = k
		seq = ''
		for k in np.argmax(mat,1):
			seq+=rev_dict[k]
		return seq

	def decode_vec2seq(self,vec):
		rev_dict = {}	
		for k,v in self.pos_dict.items():
			rev_dict[v] = k
		seq = ''
		for k in vec:
			seq+=rev_dict[k]
		return seq

	def build_dataset(self,addr,X_encode='onehot',Y_encode = 'onesided'):
		#build dataset
		try:
			CWDIR = os.path.abspath(os.path.dirname(__file__))
		except:
			CWDIR = os.getcwd()
		df = pd.read_csv(addr)
		if Y_encode == 'onesided':
			df['label'] = np.where(df['label'].astype('f') == 1.0, 1,.1)
		else:
			df['label'] = np.where(df['label'].astype('f') == 1.0, 1., 0.)

		Y = df['label'].values
		if (X_encode == 'onehot') | (X_encode == 'embedding'):
			encode = self.encode_seq2mat
		elif (X_encode == 'integer') : #embeddng is not used in prepare_data but to be syn with go_CNN and go_GAN
			encode = self.encode_seq2vec		
		X_values = df['ecfp'].values
		X = np.array([encode(X_values[i]) for i in range(len(df))])
		return X,Y


	def get_data(self,trainData_id,valData_id,testData_id):
		# a function that input id and get data

		try:
			CWDIR = os.path.abspath(os.path.dirname(__file__))
		except:
			CWDIR = os.getcwd()
		df_menuVal= pd.read_csv(os.path.join(CWDIR,'./../../experiments/menu_valData.csv')).astype(str)
		df_menuTest= pd.read_csv(os.path.join(CWDIR,'./../../experiments/menu_testData.csv')).astype(str)
		df_menuTrain= pd.read_csv(os.path.join(CWDIR,'./../../experiments/menu_trainData.csv')).astype(str)

		
		addr_train= os.path.join(CWDIR,'./../../',df_menuTrain.loc[df_menuTrain['trainData_id']==str(trainData_id),'addr'].values[0])

		addr_val = os.path.join(CWDIR,'./../../',df_menuVal.loc[df_menuVal['valData_id']==str(valData_id),'addr'].values[0])

		addr_test = os.path.join(CWDIR,'./../../',df_menuTest.loc[df_menuTest['testData_id']==str(testData_id),'addr'].values[0])
		#get data
		X_train,Y_train = self.build_dataset(addr_train,X_encode=self.X_encode,Y_encode=self.Y_encode)
		X_val,Y_val = self.build_dataset(addr_val,X_encode=self.X_encode,Y_encode=self.Y_encode)
		X_test,Y_test = self.build_dataset(addr_test,X_encode=self.X_encode,Y_encode=self.Y_encode)
		return X_train,Y_train,X_val,Y_val,X_test,Y_test

	def generate_data(self,G,noises,X_encode='onehot',is_generating_stochastic=True):
		#it defines the logic of generating X of different shape based on a GAN generator and noises function
		def get_probs(G,noises):
			X_fake = G.predict(noises)
			probs = X_fake.copy()
			if len(self.pos_dict) >2:
				cutoff = np.percentile(X_fake,0,axis=len(X_fake.shape)-1,keepdims=True)
				probs[probs<cutoff] = 0
				sums = probs.sum(axis=len(X_fake.shape)-1,keepdims=True)
				probs = np.squeeze(probs/sums)
				print('normalized')
			return probs
		def convert_vec2mat(vec,n_choices):
			mat = np.zeros(n_choices)
			for i in range(len(vec)):
				mat[i,vec[i]] = 1
			return mat

		def generate_sequence(prob,X_encode=X_encode):
	#		vec = np.array([np.random.choice(range(prob.shape[1]), size = 1, p = prob[j])[0] for j in range(prob.shape[0])])
			npr = np.random.RandomState(123)
			if len(self.pos_dict) ==2:
				vec = np.array([npr.choice((1,0), size = 1, p = (prob[i],1-prob[i]))[0] for i in range(prob.shape[0])])
			elif len(self.pos_dict) >2:
				vec = np.array([npr.choice(range(prob.shape[1]), size = 1, p = prob[j])[0] for j in range(prob.shape[0])])
			if X_encode == 'integer':
				return vec
			elif X_encode == 'onehot':
				return convert_vec2mat(vec,n_choices=prob.shape)

		probs = get_probs(G=G,noises=noises)
#		print(probs)
		num = self.scales/probs[0].shape[0] 
		
		if is_generating_stochastic == True:
			scaling_number = int(noises.shape[0]/num)
			if scaling_number <1:
				scaling_number = 1
			outs = multi_work(thelist=list(enumerate(list(probs))),func=generate_sequence,arguments=[[X_encode]],scaling_number=scaling_number,on_disk=False)
		elif is_generating_stochastic == False:
			if len(self.pos_dict) ==2:
				outs = (probs>0.5).astype(int)
			elif len(self.pos_dict) !=2:
				outs=[convert_vec2mat(vec,n_choices=probs[0].shape) for vec in np.argmax(probs,2)]

		return np.array(outs)


if __name__ == '__main__':
	pdata=prepare_data(X_encode = 'onehot',Y_encode = 'integer')
	trainData_id = 0
	valData_id = 0
	testData_id = 0
	X_train,Y_train,X_val,Y_val,X_test,Y_test = pdata.get_data(trainData_id,valData_id,testData_id)