from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.pyplot import imread
from keras import optimizers
import model 
import img_utils
import loses
import scipy.io as sio

# python main_program.py {0,1,2,3} {1,2,3}
IMAGE_ORDERING = 'channels_last' 
images_path='../data/input/'
#segs_path='../new_data/mask'+sys.argv[2]+'/'
segs_path='../data/output/mask'+sys.argv[2]+'/'
batch_size=len(os.listdir(images_path))
#n_classes=int(sys.argv[3])
n_classes = 2
height=68
width=68
cv=int(sys.argv[1]);

actual_set=['F1','F2','M1','M2']
actual_set = np.roll(actual_set,cv+1)
train_matrix= actual_set[1:4]
test_matrix=actual_set[0:1]
print(actual_set)
print(train_matrix)
print(test_matrix)

directory='./pred_FCN/'+actual_set[0]+'/mask'+sys.argv[2];
#epoch=int(sys.argv[4])
epoch= 50
new = 2
#new=int(sys.argv[5])
print('**************************************************************')
print('seg_path : '+segs_path)
print('number of classes : '+str(n_classes))
print('directory : '+directory)
print('epochs : '+str(epoch))
print('Cross Validation :'+ str(cv))
print('**************************************************************')

if not os.path.exists(directory):
    print("creating directory ")
    os.makedirs(directory)
    train_flag=1;
    
else:
    print("directory already exists")
    train_flag=1;	
    
#start=int(sys.argv[1]);
'''
print('************ Cross fold : '+str(int((start+4)/4)))
actual_set=np.array([354,355,356,357,358,359,360,361,362,382,386,387,389,390,394,395])
actual_set=np.roll(actual_set,-1*start)
train_matrix=actual_set[0:8];
dev_matrix=actual_set[8:12];
test_matrix=actual_set[12:16];
'''

# make changes 
X_train,Y_train,X_test,Y_test,name=img_utils.imageSegmentationGenerator_subind(images_path , segs_path ,  batch_size,  n_classes , height , width, test_matrix,train_matrix)


if(new==1):
	print('*********new model fitting*********')
	segnet_model=model.FCN8(n_classes ,68, 68)
	#segnet_model.summary()
	model_path = directory+'/model_'+actual_set[0];
	callbacks = [EarlyStopping(monitor='val_acc', patience=4, min_delta = 0.0001), ModelCheckpoint(model_path, monitor='val_acc', verbose=0, save_best_only=True, mode='auto', period=1)]
	segnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	history=segnet_model.fit(X_train, Y_train,epochs=50,batch_size=32,validation_data=(X_test,Y_test),verbose=1, callbacks=callbacks)

#############################################################
# Save the model not the weights 
#############################################################
	#fcn_model.save_weights(directory+'/model_weight',overwrite=True)
	print('************ path : '+directory+'/model_'+actual_set[0])
	segnet_model.save(directory+'/model_'+actual_set[0])
else:
	print('*********old model fitting*********')
	print('************ path : '+directory+'/model_'+actual_set[0])
	segnet_model = load_model(directory+'/model_'+actual_set[0])
	segnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        history=segnet_model.fit(X_train, Y_train,epochs=20,batch_size=32,validation_data=(X_test,Y_test),verbose=1)
	print('************ path : '+directory+'/model_'+actual_set[0])
        segnet_model.save(directory+'/model_'+actual_set[0])
	#scores = segnet_model.evaluate(X_test, Y_test, verbose=0)
	#print(scores)
	#print('**************** Score : '+str(scores)+' *****************')



#############################################################
# Predictions
#############################################################
test_path=directory+'/test_'+actual_set[0]
# dev_path=directory+'/dev_cv'+str(int(start/4))
if not os.path.exists(test_path):
    print("creating test directory : "+test_path)
    os.makedirs(test_path)
else:
    print("image directory already exists")

img_utils.predicted_output(segnet_model,X_test,name,test_path, height,width,n_classes)

'''
if not os.path.exists(dev_path):
    print("creating development directory : "+dev_path)
    os.makedirs(dev_path)
else:
    print("image directory already exists")

img_utils.predicted_output(segnet_model,X_dev,name_dev,dev_path, height,width,n_classes)
'''