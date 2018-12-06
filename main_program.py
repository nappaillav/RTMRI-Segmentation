from keras.models import *
from keras.layers import *
import cv2
import glob
import numpy as np
#import matplotlib.pyplot as plt
import os
import sys
#from matplotlib.pyplot import imread
from keras import optimizers
import model 
import utils
import model
import scipy.io as sio
from keras.callbacks import ModelCheckpoint,EarlyStopping
 
#directory for predicted model 
directory = './pred_seq'

list_score_per_frames=[]
# mask to be choosen
mask = int(sys.argv[1])
# number of train images
#num = int(sys.argv[2])

# read images from all subject and create
print("input and output dimensions") 
# Xtrain, Ytrain, Xtest, Ytest, Xdev,Ydev,test_name = utils.readImage(['F1','F2','M1','M2'], mask)

X_F1, Y_F1, list_F1 = utils.readImage_sub('F1',mask)
X_F2, Y_F2, list_F2 = utils.readImage_sub('F2',mask)
X_M1, Y_M1, list_M1 = utils.readImage_sub('M1',mask)
X_M2, Y_M2, list_M2 = utils.readImage_sub('M2',mask)
keys = X_F1.keys() 
keys = np.sort(keys)

num_range = np.array([1,2,3,4,5,6,7,8])

# generate test set
Xtest = []
Ytest = []
for it in range(14,18):
	print('Test video : '+keys[it])
	new_X = X_F1[keys[it]]+X_F2[keys[it]]+X_M1[keys[it]]+X_M2[keys[it]]
	new_Y = Y_F1[keys[it]]+Y_F2[keys[it]]+Y_M1[keys[it]]+Y_M2[keys[it]]
	Xtest += new_X
	Ytest += new_Y

Xtest = np.array(Xtest)
Ytest = np.array(Ytest)

Xdev = []
Ydev = []
for it in range(12,13):
	print('Devlopment video : '+keys[it])
	new_X = X_F1[keys[it]]+X_F2[keys[it]]+X_M1[keys[it]]+X_M2[keys[it]]
	new_Y = Y_F1[keys[it]]+Y_F2[keys[it]]+Y_M1[keys[it]]+Y_M2[keys[it]]
	Xdev += new_X
	Ydev += new_Y

Xdev = np.array(Xdev)
Ydev = np.array(Ydev)

print('Dimension of X_test: '+str(Xtest.shape))
print('Dimension of Y_test: '+str(Ytest.shape))

print('Dimension of Xdev: '+str(Xdev.shape))
print('Dimension of Ydev: '+str(Ydev.shape))

for i in num_range:
	# model initialization
	# X = np.vstack((Xtrain['F1'][0:i],Xtrain['F2'][0:i],Xtrain['M1'][0:i],Xtrain['M2'][0:i]))
	# Y = np.vstack((Ytrain['F1'][0:i],Ytrain['F2'][0:i],Ytrain['M1'][0:i],Ytrain['M2'][0:i]))
	# Training Data Set
	X= []
	Y= []
	for it in range(0,i):
		print('Training video: '+keys[it])
		new_X = X_F1[keys[it]]+X_F2[keys[it]]+X_M1[keys[it]]+X_M2[keys[it]]
		new_Y = Y_F1[keys[it]]+Y_F2[keys[it]]+Y_M1[keys[it]]+Y_M2[keys[it]]
		X += new_X
		Y += new_Y

	X = np.array(X)
	Y = np.array(Y)
	Segnet_model = model.Segnet();
	#print('output shape : ' + str(Segnet_model.outputHeight)+'x'+str(Segnet_model.outputHeight))
	print('number of Frames: '+str(len(X)))
	print('Dimension of Train: '+str(X.shape))

	### Model path #### 
	model_path = directory+'/best_m_'+str(mask)+'_v_'+str(i);
	Segnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
	saveBestModel = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
	
	history=Segnet_model.fit(X, Y, epochs=70,batch_size=16,validation_data=(Xtest,Ytest),callbacks=[earlyStopping, saveBestModel],verbose=1)
	#history=Segnet_model.fit(X, Y, epochs=50,batch_size=16,validation_data=(Xdev,Ydev),verbose=1)	
	#Segnet_model.save(directory+'/mask'+str(mask)+'/model_'+str(i))
	
	### test images ####
	'''
	test_path=directory+'/mask'+str(mask)+'/'+str(i)
	Segnet_model.load_weights(model_path)
	if not os.path.exists(test_path):
        	print("creating test directory : "+test_path)
        	os.makedirs(test_path)
	else:
		print("image directory already exists")

	utils.predicted_output(Segnet_model,Xtest,test_name,test_path, 68,68,2)
	'''
	### prints the scores for each range ####
	print('loading the best model...')
	Segnet_model.load_weights(model_path)	
	scores = Segnet_model.evaluate(Xtest, Ytest, verbose=0)
	print('scores : '+str(scores))
	#new_scores = (scores*16384 - 11760)/4624
	#print('Actual scores : '+str(new_scores))
	test_sample = []
	test_sample = [i,mask,scores[0],scores[1]]
	list_score_per_frames.append(test_sample)


np.savetxt(directory+'/scores_'+str(mask)+'.txt', np.array(list_score_per_frames),fmt = '%5.6f')
print(list_score_per_frames)



