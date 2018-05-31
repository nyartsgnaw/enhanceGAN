from ggplot import *
import numpy as np 
import pandas as pd
import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()
from keras.models import Model, Input
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
#import_local_package(os.path.join(CWDIR,'./../../lib/utils/utils.py'),['load_model'])
from keras.models import  load_model

import_local_package(os.path.join(CWDIR,'./../../data/lib/prepare_data.py'),['prepare_data'])
import_local_package(os.path.join(CWDIR,'./../../lib/utils/train_GAN.py'),['train_GAN'])
import_local_package(os.path.join(CWDIR,'./../../experiments/models/GAN_2.py'),['create_T','create_D','create_G'])

def get_layeroutput(model,data,layer_name):
	#evaluate keras output layers values by layer name
	intermediate_layer_model = Model(inputs=model.get_input_at(0),
									outputs=model.get_layer(layer_name).output)
	intermediate_output = intermediate_layer_model.predict(data)
	return intermediate_output


def PCA_plot(X,y,title='network output'):
	#title: title for the output plot

	feat_cols = ['p'+str(i) for i in range(X.shape[1]) ]
	df = pd.DataFrame(X,columns=feat_cols)
	df['label'] = y
	from sklearn.decomposition import PCA

	pca = PCA(n_components=2)
	pca_result = pca.fit_transform(df[feat_cols].values)

	df['pca-one'] = pca_result[:,0]
	df['pca-two'] = pca_result[:,1] 
	#df['pca-three'] = pca_result[:,2]
	rndperm = np.random.permutation(df.shape[0])

	print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


	chart = ggplot( df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-two', color='label') ) \
			+ geom_point(size=10,alpha=0.95) \
			+ ggtitle('{}'.format(title))
	return chart


def tSNE_plot(X,y,title='network output'):
	#title: title for the output plot

	feat_cols = [ 'p'+str(i) for i in range(X.shape[1]) ]
	df = pd.DataFrame(X,columns=feat_cols)
	df['label'] = y

	from sklearn.manifold import TSNE

	n_sne = 7000
	rndperm = np.random.permutation(df.shape[0])

	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

	df_tsne = df.loc[rndperm[:n_sne],:].copy()
	df_tsne['tsne-one'] = tsne_results[:,0]
	df_tsne['tsne-two'] = tsne_results[:,1]

	chart = ggplot( df_tsne, aes(x='tsne-one', y='tsne-two', color='label') ) \
			+ geom_point(size=10,alpha=0.95) \
			+ ggtitle('{}'.format(title))
	return chart


class plot_GAN_outlayers(object):
	# a class that generates tSNE and PCA plots for discriminator, tester output space of experiments
	def __init__(self,exp,plot_PCA=True,plot_tSNE=False):
		self.plot_PCA = plot_PCA # whether to plot PCA
		self.plot_tSNE = plot_tSNE #whether to plot tSNE
		self.exp = exp #experiment setups
#		self.T = load_model(os.path.join(CWDIR,'./../../logs/trained_models/tester/GAN_T_exp{}.h5'.format(exp['exp_id'])))
		df_exp_log_BT = pd.read_excel(os.path.join(CWDIR,'./../compare/testGAN/exp_logs_TB0.xlsx'))
		self.exp_id_B = df_exp_log_BT.loc[df_exp_log_BT['trainData_id']==exp['trainData_id'],'exp_id'].values[0]
		self.testData_id_BT = df_exp_log_BT.loc[df_exp_log_BT['trainData_id']==exp['trainData_id'],'testData_id'].values[0]
		df_exp_log_GT = pd.read_excel(os.path.join(CWDIR,'./../compare/testGAN/exp_logs_TG0.xlsx'))
		self.exp_id_G = df_exp_log_GT.loc[df_exp_log_GT['trainData_id']==exp['trainData_id'],'exp_id'].values[0]
		self.testData_id_GT = df_exp_log_GT.loc[df_exp_log_GT['trainData_id']==exp['trainData_id'],'testData_id'].values[0]
		
		self.D = load_model(os.path.join(CWDIR,'./../../logs/trained_models/GAN/GAN_D_exp{}.h5'.format(exp['exp_id'])))
#		D = create_D()
#		self.D = load_model(D,os.path.join(CWDIR,'./../../logs/trained_models/GAN/GAN_D_exp{}.h5'.format(exp['exp_id'])))
		self.G = load_model(os.path.join(CWDIR,'./../../logs/trained_models/GAN/GAN_G_exp{}.h5'.format(exp['exp_id'])))
#		G = create_G()
#		self.G = load_model(G,os.path.join(CWDIR,'./../../logs/trained_models/GAN/GAN_G_exp{}.h5'.format(exp['exp_id'])))
		self.pdata=prepare_data(X_encode = 'onehot',Y_encode = 'integer')

		self.GAN = train_GAN(
					X_encode = 'onehot',\
					Y_encode = 'integer',\
					exp_id = '0',\
					D_class_weight={'true':1.2,'fake':1,'false':1.2},\
					G_class_weight={'true':1,'fake':1,'false':1},\
					is_generating_stochastic = False)

	def plot_reduced_dim(self,train_data,CV_type='train'):

		train_data['X_fake'] = self.pdata.generate_data(self.G,train_data['X_noise'],X_encode='onehot',is_generating_stochastic=True)
	
		layer_names = {i:self.tester.layers[i].name for i in range(len(self.tester.layers))}

		import re
		
		X_layers = {}
		i_dense0 = [i for i,x in layer_names.items() if re.search('dense',x)][0]

		for i in range(i_dense0,i_dense0+2):
			X_layers['L'+str(i)+'-'+layer_names[i]] = {k:get_layeroutput(self.tester,train_data[k],layer_names[i]) for k in ['X_true','X_fake','X_false']}
			X_seq = {}
			for k in ['X_true','X_fake','X_false']:
				X = []
				for x in train_data[k]:
					X.append(self.pdata.encode_seq2vec(self.pdata.decode_mat2seq(x)))
				X_seq[k] = np.array(X)
			X_layers['data'] = X_seq
		
		if self.plot_PCA ==True:
			plot_PCA = {}
			for layer in X_layers.keys():
				train_data = X_layers[layer]
				X = np.concatenate([train_data['X_true'],\
									train_data['X_fake'],\
									train_data['X_false']])
				y = np.concatenate([['X_true']*train_data['X_true'].shape[0],\
									['X_fake']*train_data['X_fake'].shape[0],\
									['X_false']*train_data['X_false'].shape[0]])				
				
				plot_PCA[layer] = PCA_plot(X,y,title='PCA-reduced {} output @layer: {}\n CV_type: {}, exp_id: {}'.format(self.tester_name,layer,CV_type,self.exp['exp_id']))
				plot_PCA[layer].save(os.path.join(CWDIR,'./../../analysis/reduce_dim/plots/PCA/{}/exp{}_PCA_{}_{}.jpg'.format(self.tester_name,self.exp['exp_id'],CV_type,layer)))
				
		if self.plot_tSNE == True:
			plot_tSNE = {}
			for layer in X_layers.keys():
				train_data = X_layers[layer]
				X = np.concatenate([train_data['X_true'],\
									train_data['X_fake'],\
									train_data['X_false']])
				y = np.concatenate([['X_true']*train_data['X_true'].shape[0],\
									['X_fake']*train_data['X_fake'].shape[0],\
									['X_false']*train_data['X_false'].shape[0]])					
				plot_tSNE[layer] = tSNE_plot(X,y,title='tSNE-reduced {} output @layer: {}\n CV_type: {}, exp_id: {}'.format(self.tester_name,layer,CV_type,self.exp['exp_id']))
				plot_tSNE[layer].save(os.path.join(CWDIR,'./../../analysis/reduce_dim/plots/tSNE/{}/exp{}_tSNE_{}_{}.jpg'.format(self.tester_name,self.exp['exp_id'],CV_type,layer)))
		return 

	def start_plot(self,tester_type='B'):  
#		T = create_T()
		if tester_type == 'B':
			self.tester = load_model(os.path.join(CWDIR,'./../compare/testGAN/logs/trained_models/TB_{}.h5'.format(self.exp_id_B)))
#			self.tester = load_model(T,os.path.join(CWDIR,'./../compare/testGAN/logs/trained_models/TB_{}.h5'.format(self.exp_id_B)))
			self.tester_name = 'blank_tester'
			testData_id = self.testData_id_BT

		elif tester_type =='G':
			self.tester = load_model(os.path.join(CWDIR,'./../compare/testGAN/logs/trained_models/TG_{}.h5'.format(self.exp_id_G)))
#			self.tester = load_model(T,os.path.join(CWDIR,'./../compare/testGAN/logs/trained_models/TG_{}.h5'.format(self.exp_id_G)))
			self.tester_name = 'GAN_tester'			
			testData_id = self.testData_id_BT

		elif tester_type == 'D':
			self.tester = self.D
			self.tester_name = 'discriminator'	
			testData_id = self.testData_id_BT
		
		os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./../../analysis/reduce_dim/plots/PCA/{}/').format(self.tester_name)))

		X_train,Y_train,X_val,Y_val,X_test,Y_test = self.pdata.get_data(self.exp['trainData_id'],self.exp['valData_id'],testData_id)

		train_data = self.GAN.init_GAN_trainingData(X_train,Y_train,G_input_dim=100)
		self.plot_reduced_dim(train_data,'train')

		train_data = self.GAN.init_GAN_trainingData(X_val,Y_val,G_input_dim=100)
		self.plot_reduced_dim(train_data,'val')
		train_data = self.GAN.init_GAN_trainingData(X_test,Y_test,G_input_dim=100)
		self.plot_reduced_dim(train_data,'test')
		return 


if __name__ == '__main__':
	exp_addr = os.path.join(CWDIR,'./../../exp_logs.xlsx')
	df = pd.read_excel(exp_addr)
	for i in range(df.shape[0]):
#		if df.iloc[i]['n_tested'] == 0:
#			try:
			exp = df.iloc[i]
			pgo = plot_GAN_outlayers(exp,plot_PCA=True,plot_tSNE=False)
			pgo.start_plot(tester_type ='B')
			pgo.start_plot(tester_type = 'D')
#			except Exception as e:
#				print(e)

	 	 