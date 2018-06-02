#the file that describe generator,discriminator and tester structures
#generally speaking, a shared tester structure is recommended
from keras.layers.convolutional import Conv2DTranspose , Conv1D, Conv2D,Convolution3D, MaxPooling2D,UpSampling1D,UpSampling2D,UpSampling3D
from keras.layers import Input,Embedding, Dense, Dropout, Activation, Flatten,   Reshape, Flatten, Lambda
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import regularizers
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
import numpy as np 
import pandas as pd
import os

def create_G(input_dim=(100,),output_dim=(9,20)):
	G = Sequential()
	G.add(Dense(input_shape=input_dim, \
					units= 128, \
					kernel_initializer=initializers.random_normal(stddev=0.02)))

	G.add(BatchNormalization())
	#G.add(Conv2DTranspose(32, 5, strides=(2,1), activation=Activation('relu'), padding='same',kernel_initializer='glorot_uniform'))
#    G.add(GaussianDropout(0.25))  #https://arxiv.org/pdf/1611.07004v1.pdf
#    G.add(GaussianNoise(0.05))
	G.add(Activation('relu'))

	G.add(Dense(128))
	G.add(BatchNormalization())
	G.add(GaussianDropout(0.25))  #https://arxiv.org/pdf/1611.07004v1.pdf
	G.add(GaussianNoise(0.05))
	G.add(Activation('relu'))

	G.add(Dense(np.prod(output_dim),kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
	G.add(BatchNormalization())
#    G.add(GaussianDropout(0.25))  #https://arxiv.org/pdf/1611.07004v1.pdf
#    G.add(GaussianNoise(0.05))

	G.add(Activation('sigmoid'))

	G.add(Reshape(output_dim))
	return(G)

def create_D(input_dim=(9,20),output_dim=3):
	D = Sequential()
	D.add(Dense(input_shape=input_dim, \
				units= 128, \
				kernel_initializer=initializers.random_normal(stddev=0.02)))
	D.add(BatchNormalization())
	#G.add(Conv2DTranspose(32, 5, strides=(2,1), activation=Activation('relu'), padding='same',kernel_initializer='glorot_uniform'))
#    D.add(GaussianDropout(0.25))  #https://arxiv.org/pdf/1611.07004v1.pdf
#    D.add(GaussianNoise(0.05))
	D.add(Activation('relu'))

#    D.add(Dense(128,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
	D.add(Dense(128))
	D.add(BatchNormalization())
	D.add(Activation('relu'))

#    D.add(Dense(128,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
	D.add(Dense(128))
	D.add(BatchNormalization())
	D.add(Activation('relu'))
	D.add(Dense(output_dim))
	D.add(Activation('sigmoid'))

	return(D)

def create_T(input_dim=(9,20),output_dim=1):
	D = Sequential()
	D.add(Dense(input_shape=input_dim, \
				units= 128, \
				kernel_initializer=initializers.random_normal(stddev=0.02)))
	D.add(BatchNormalization())
	#G.add(Conv2DTranspose(32, 5, strides=(2,1), activation=Activation('relu'), padding='same',kernel_initializer='glorot_uniform'))
#    D.add(GaussianDropout(0.25))  #https://arxiv.org/pdf/1611.07004v1.pdf
#    D.add(GaussianNoise(0.05))
	D.add(Activation('relu'))

#    D.add(Dense(128,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
	D.add(Dense(128))
	D.add(BatchNormalization())
	D.add(Activation('relu'))

#    D.add(Dense(128,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
	D.add(Dense(128))
	D.add(BatchNormalization())
	D.add(Activation('relu'))
	D.add(Dense(output_dim))
	D.add(Activation('sigmoid'))
	return(D)


if __name__ == '__main__':
	model_id = 'GAN_2'
	G = create_G()
	D = create_D()
	T = create_T()
	df_menu = pd.read_csv('./../../experiments/menu_models.csv')
	df_menu.loc[df_menu['model_id']==model_id,'D_input_dim'] = str(D.input_shape)
	df_menu.loc[df_menu['model_id']==model_id,'G_input_dim'] = str(G.input_shape)
	df_menu.loc[df_menu['model_id']==model_id,'T_input_dim'] = str(T.input_shape)
	df_menu.loc[df_menu['model_id']==model_id,'D_output_dim'] = str(D.output_shape)
	df_menu.loc[df_menu['model_id']==model_id,'G_output_dim'] = str(G.output_shape)
	df_menu.loc[df_menu['model_id']==model_id,'T_output_dim'] = str(T.output_shape)
	df_menu.to_csv('./../../experiments/menu_models.csv',index=False)
