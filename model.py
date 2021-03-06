import numpy as np 
#import pandas as pd
import os
from keras.models import *
from keras.layers import *
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.models import Model, Sequential

VGG_Weights_path = './vgg16_weights_th_dim_ordering_th_kernels.h5'

def VGGSegnet( n_classes=2 ,  input_height=128, input_width=128 , vgg_level=3):

    img_input = Input(shape=(input_height,input_width,3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last' )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_last' )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_last' )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_last' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_last' )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_last' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_last' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_last' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_last' )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_last' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_last' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_last' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_last' )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_last' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_last' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_last' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_last' )(x)
    f5 = x

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense( 1000 , activation='softmax', name='predictions')(x)

    vgg  = Model(  img_input , x  )
    #vgg.load_weights(VGG_Weights_path); print('Imagenet weights restored')

    levels = [f1 , f2 , f3 , f4 , f5 ]

    o = levels[ vgg_level ]
    
    o = ( UpSampling2D((2,2)  , data_format='channels_last' ))(o)
    o = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = ( ZeroPadding2D( (1,1), data_format='channels_last'))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_last' ) )(o)
    o = ( ZeroPadding2D((1,1) , data_format='channels_last' ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_last' ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_last' ))(o)
    o = ( ZeroPadding2D((1,1)  , data_format='channels_last' ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_last' ))(o)
    o = ( BatchNormalization())(o)


    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_last' )( o )
    o_shape = Model(img_input , o ).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model

def VGG19Segnet(n_classes=2 ,  input_height=68, input_width=68 , vgg_level=3):
	print('VGG16 Loading..')
	base_model = VGG16(input_shape =(68, 68, 3), weights='imagenet', include_top = False)
	o = ( UpSampling2D((2,2)  , data_format='channels_last' ))(base_model.get_layer('block4_pool').output)
	o = ( Conv2D(512, (3, 3), padding='same', data_format='channels_last'))(o)
	o = (Add())([ o , base_model.get_layer('block4_conv3').output ])
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)
	o = ( Conv2D(512, (3, 3), padding='same', data_format='channels_last'))(o)
	o = (Add())([ o , base_model.get_layer('block4_conv2').output ])
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)
	o = ( Conv2D(512, (3, 3), padding='same', data_format='channels_last'))(o)
	o = (Add())([ o , base_model.get_layer('block4_conv1').output ])
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)

	#o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
	o = Conv2DTranspose( 512 , kernel_size=(3,3) ,  strides=(2,2))(o)
	#o = ( ZeroPadding2D( (1,1), data_format='channels_last'))(o)
	o = ( Conv2D( 512, (3, 3), padding='same', data_format='channels_last'))(o)
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)
	o = ( Conv2D( 512, (3, 3), padding='same', data_format='channels_last'))(o)
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)
	o = ( Conv2D( 256, (3, 3), padding='same', data_format='channels_last'))(o)
	#o = (Add())([ o , base_model.get_layer('block3_conv4').output ])
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)

	o = ( UpSampling2D((2,2)  , data_format='channels_last' ) )(o)
	o = ( Conv2D( 256 , (3, 3), padding='same' , data_format='channels_last' ))(o)
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)
	o = ( Conv2D( 256 , (3, 3), padding='same' , data_format='channels_last' ))(o)
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)
	o = ( Conv2D( 128 , (3, 3), padding='same' , data_format='channels_last' ))(o)
	#o = (Add())([ o , base_model.get_layer('block2_conv2').output ])
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)

	o = ( UpSampling2D((2,2)  , data_format='channels_last' ))(o)
	o = ( Conv2D( 128 , (3, 3), padding='same'  , data_format='channels_last' ))(o)
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)
	o = ( Conv2D( 64 , (3, 3), padding='same'  , data_format='channels_last' ))(o)
	#o = (Add())([ o , base_model.get_layer('block1_conv2').output ])
	o = ( BatchNormalization())(o)
	o = (Activation('relu'))(o)


	o =  Conv2D(n_classes , (3, 3) , padding='same', data_format='channels_last' )( o )
	o = ( BatchNormalization())(o)
	o_shape = Model(base_model.input , o ).output_shape
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model( base_model.input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight
	return model

def CreateSegNet(n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax"):
	# encoder
	inputs = Input(shape=(68,68,3))

	conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
	conv_1 = BatchNormalization()(conv_1)
	conv_1 = Activation("relu")(conv_1)
	conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
	conv_2 = BatchNormalization()(conv_2)
	conv_2 = Activation("relu")(conv_2)

	pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

	conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
	conv_3 = BatchNormalization()(conv_3)
	conv_3 = Activation("relu")(conv_3)
	conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
	conv_4 = BatchNormalization()(conv_4)
	conv_4 = Activation("relu")(conv_4)

	pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

	conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
	conv_5 = BatchNormalization()(conv_5)
	conv_5 = Activation("relu")(conv_5)
	conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
	conv_6 = BatchNormalization()(conv_6)
	conv_6 = Activation("relu")(conv_6)
	conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
	conv_7 = BatchNormalization()(conv_7)
	conv_7 = Activation("relu")(conv_7)

	pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

	conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
	conv_8 = BatchNormalization()(conv_8)
	conv_8 = Activation("relu")(conv_8)
	conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
	conv_9 = BatchNormalization()(conv_9)
	conv_9 = Activation("relu")(conv_9)
	conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
	conv_10 = BatchNormalization()(conv_10)
	conv_10 = Activation("relu")(conv_10)

	pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

	conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
	conv_11 = BatchNormalization()(conv_11)
	conv_11 = Activation("relu")(conv_11)
	conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
	conv_12 = BatchNormalization()(conv_12)
	conv_12 = Activation("relu")(conv_12)
	conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
	conv_13 = BatchNormalization()(conv_13)
	conv_13 = Activation("relu")(conv_13)

	pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
	print("Build enceder done..")

	# decoder

	unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

	conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
	conv_14 = BatchNormalization()(conv_14)
	conv_14 = Activation("relu")(conv_14)
	conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
	conv_15 = BatchNormalization()(conv_15)
	conv_15 = Activation("relu")(conv_15)
	conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
	conv_16 = BatchNormalization()(conv_16)
	conv_16 = Activation("relu")(conv_16)

	unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])
	zero_pad = ZeroPadding2D( (1,1), data_format='channels_last')(unpool_2)
	conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(zero_pad)
	conv_17 = BatchNormalization()(conv_17)
	conv_17 = Activation("relu")(conv_17)
	conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
	conv_18 = BatchNormalization()(conv_18)
	conv_18 = Activation("relu")(conv_18)
	conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
	conv_19 = BatchNormalization()(conv_19)
	conv_19 = Activation("relu")(conv_19)

	unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

	conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
	conv_20 = BatchNormalization()(conv_20)
	conv_20 = Activation("relu")(conv_20)
	conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
	conv_21 = BatchNormalization()(conv_21)
	conv_21 = Activation("relu")(conv_21)
	conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
	conv_22 = BatchNormalization()(conv_22)
	conv_22 = Activation("relu")(conv_22)

	unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

	conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
	conv_23 = BatchNormalization()(conv_23)
	conv_23 = Activation("relu")(conv_23)
	conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
	conv_24 = BatchNormalization()(conv_24)
	conv_24 = Activation("relu")(conv_24)

	unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

	conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
	conv_25 = BatchNormalization()(conv_25)
	conv_25 = Activation("relu")(conv_25)

	conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
	conv_26 = BatchNormalization()(conv_26)
	conv_26 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

	outputs = Activation(output_mode)(conv_26)
	print("Build decoder done..")

	segnet = Model(inputs=inputs, outputs=outputs, name="SegNet")

	return segnet

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

	autoencoder = Sequential()
	autoencoder.encoding_layers = encoding_layers

	for l in autoencoder.encoding_layers:
		autoencoder.add(l)
	#print(l.input_shape,l.output_shape,l)


	autoencoder.decoding_layers = decoding_layers
	for l in autoencoder.decoding_layers:
		autoencoder.add(l)
	#print(l.input_shape,l.output_shape,l)

	return autoencoder

