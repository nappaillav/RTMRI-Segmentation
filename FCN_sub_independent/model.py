from keras.models import *
from keras.layers import *
import cv2
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.pyplot import imread
from keras import optimizers
from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Conv2D
from keras.layers.normalization import BatchNormalization
import json
import theano
from theano import function, config, shared, tensor
import numpy
import time

def crop( o1 , o2 , i ,IMAGE_ORDERING='channels_last' ):
    o_shape2 = Model( i  , o2 ).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model( i  , o1 ).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]

    cx = abs( outputWidth1 - outputWidth2 )
    cy = abs( outputHeight2 - outputHeight1 )

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o2)
    if outputHeight1 > outputHeight2 :
        o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o2)
    return o1 , o2 

def FCN8( nClasses=2 ,  input_height=68, input_width=68 , vgg_level=3,IMAGE_ORDERING='channels_last'):

    img_input = Input(shape=(input_height,input_width,3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
    f5 = x
    
    o = f5

    o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING ))(o)
    o = BatchNormalization()(o)
    o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING ))(o)
    o = BatchNormalization()(o)
    

    o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING ))(o)
    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    o2 = f4
    o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)

    o , o2 = crop( o , o2 , img_input )

    o = Add()([ o , o2 ])

    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o2 = f3 
    o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)
    o2 , o = crop( o2 , o , img_input )
    o  = Add()([ o2 , o ])


    o = Conv2DTranspose( nClasses , kernel_size=(12,12) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    o_shape = Model(img_input , o ).output_shape

    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    print(o_shape)
    o = (Reshape((-1  , outputHeight*outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model

def FCN16( nClasses=2 ,  input_height=68, input_width=68 , vgg_level=3,IMAGE_ORDERING='channels_last'):

    img_input = Input(shape=(input_height,input_width,3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
    f5 = x
    
    o = f5

    o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    o2 = f4
    o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)

    o , o2 = crop( o , o2 , img_input )
    o = Add()([ o , o2 ])
    o = Conv2DTranspose( nClasses , kernel_size=(20,20) ,  strides=(16,16) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    o_shape = Model(img_input , o ).output_shape

    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    print(o_shape)
    o = (Reshape((-1  , outputHeight*outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model

def FCN32( nClasses=2 ,  input_height=68, input_width=68 , vgg_level=3,IMAGE_ORDERING='channels_last'):

    img_input = Input(shape=(input_height,input_width,3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
    f5 = x
  
    o = f5

    o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = Conv2DTranspose( nClasses , kernel_size=(36,36) ,  strides=(32,32) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    o_shape = Model(img_input , o ).output_shape

    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    print(o_shape)
    o = (Reshape((-1  , outputHeight*outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model

def Segnet( n_labels=2 ,img_h=68 , img_w=68 ):
	kernel = 3
	encoding_layers = [
		Conv2D(64, (kernel,kernel), padding='same', input_shape=( img_h, img_w,3)),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(64, (kernel,kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		MaxPooling2D(),
		# 34*34

		Conv2D(128, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(128, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		MaxPooling2D(),
		#17*17

		Conv2D(256, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(256, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(256, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		MaxPooling2D(),
		#8*8
		Conv2D(512, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(512, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(512, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		MaxPooling2D(),
		#4*4

		]



	decoding_layers = [
		UpSampling2D(),
		Conv2D(512, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(512, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(512, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),

		Conv2DTranspose( 512 , kernel_size=(3,3) ,  strides=(2,2)),
		Conv2D(512, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(512, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(256, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),

		UpSampling2D(),
		Conv2D(256, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(256, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(128, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),

		UpSampling2D(),
		Conv2D(128, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(64, (kernel, kernel), padding='same'),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(n_labels, (1, 1), padding='valid'),
		BatchNormalization(),
		Reshape((n_labels, img_h * img_w)),
		Permute((2, 1)),
		Activation('softmax'),
		]

	autoencoder = models.Sequential()
	autoencoder.encoding_layers = encoding_layers

	for l in autoencoder.encoding_layers:
		autoencoder.add(l)
	#print(l.input_shape,l.output_shape,l)


	autoencoder.decoding_layers = decoding_layers
	for l in autoencoder.decoding_layers:
		autoencoder.add(l)
	#print(l.input_shape,l.output_shape,l)

	return autoencoder
