import os
import csv
import string
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import MaxPooling2D,Conv2D
from keras.optimizers import adam, Adadelta,adagrad,SGD
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras import applications
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.metrics import binary_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt

def generate_occluded_imageset(image_pair):
	#**** to generate the occluded image set for each image pair
	#**** image_pair: a two-element list containing the file path of the two images
	#**** return: a (320*128+1)-element list containing the concatenated images (the first ele is made of the original, the rest are of occuluted)
	try:
		TF_img = Image.open(image_pair[0])
		TF_img = np.asarray(TF_img,"float64")
		target_img = Image.open(image_pair[1])
		target_img = np.asarray(target_img,"float64")
	except Exception,e:
		print(e)
	
	data = np.empty((320*256+1,256,320,3),dtype="float64")
	data[0,0:128,:,:] = TF_img
	data[0,128:256,:,:] = target_img

	occluded_size = 16
	cnt = 1
	for i in range(256):
		for j in range(320):
			i_min = int(i - occluded_size/2)
			i_max = int(i + occluded_size/2)
			j_min = int(j - occluded_size/2)
			j_max = int(j + occluded_size/2)
			if i_min<0:
				i_min = 0
			if i_max>256:
				i_max = 256
			if j_min<0:
				j_min = 0
			if j_max>320:
				j_max = 320
			'''
			occluded_tf_img = TF_img
			occluded_tf_img[i_min:i_max,j_min:j_max,:] = 255
			occluded_tar_img = target_img
			occluded_tar_img[i_min:i_max,j_min:j_max,:] = 255
			'''
			data[cnt,0:128,:,:] = TF_img
			data[cnt,128:256,:,:] = target_img
			data[cnt,i_min:i_max,j_min:j_max,:] = 255
			cnt += 1
	
	'''
	# 8*20 blocks
	data = np.empty((128*320+1,256,320,3),dtype="float64")
	data[0,0:128,:,:] = TF_img
	data[0,128:256,:,:] = target_img

	occluded_size = 16
	cnt = 1
	for i in range(128):
		for j in range(320):
			occluded_tf_img = TF_img
			occluded_tf_img[(i*occluded_size):((i+1)*occluded_size),(j*occluded_size):((j+1)*occluded_size),:] = 255
			occluded_tar_img = target_img
			occluded_tar_img[(i*occluded_size):((i+1)*occluded_size),(j*occluded_size):((j+1)*occluded_size),:] = 255
			data[cnt,0:128,:,:] = occluded_tf_img
			data[cnt,128:256,:,:] = occluded_tar_img
			cnt += 1
	'''

	return data


def predict_model():
	#**** to define the model and load weights
	base_model = applications.ResNet50(weights=None,include_top=False,input_shape=(256, 320, 3))

	add_model = Sequential()
	add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
	add_model.add(Dense(128,activation='tanh'))
	add_model.add(Dropout(0.3))
	add_model.add(Dense(1,activation='sigmoid'))
	#add_model.load_weights('bottleneck_fully_connected_layer_model_for_resnet.h5')

	model = Model(inputs=base_model.input,outputs=add_model(base_model.output))

	sgd = SGD(lr = 1e-3, decay = 1e-5,momentum = 0.8,nesterov=True)
	model.compile(loss = 'binary_crossentropy', optimizer= sgd ,metrics = ['accuracy'])

	model.load_weights('Weight_deposit/ResNet-transferlearning2_2018_4_8_3.model')
	model.summary()

	return model


def generate_occlusion_map(predic_result):
	#**** to generate the occlusion map and save it
	#**** input predic_result: the list of predicting probablities of occluded image set by the model
	#**** output occlusion_map: the generated occlusion map
	occ_map = np.empty((256,320),dtype="float64")
	cnt = 1
	for i in range(256):
		for j in range(320):
			print(predic_result[0],predic_result[cnt])
			occ_map[i,j] = predic_result[0] - predic_result[cnt]
			cnt += 1

	np.savetxt('occlusion_map.txt',occ_map,fmt='%0.8f')

	return occ_map


#********************************************************************
#**** main program
image_pair = ["Occlusion_test/dataset/sob_lateral.bmp","Occlusion_test/dataset/Doc3_lateral.bmp"]
data = generate_occluded_imageset(image_pair)
print('occluded imageset generated!')
model = predict_model()
print('start predicting...')
predic_result = model.predict(data)
generate_occlusion_map(predic_result)
print('occlusion map generated!')

