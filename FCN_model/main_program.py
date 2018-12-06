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
import model 
import img_utils
import loses
import scipy.io as sio

# python main_program.py {0,4,8,12} {1,2,3}
IMAGE_ORDERING = 'channels_last' 
images_path='../data/input/'
#segs_path='../new_data/mask'+sys.argv[2]+'/'
segs_path='../data/output/mask'+sys.argv[2]+'/'
batch_size=len(os.listdir(images_path))
#n_classes=int(sys.argv[3])
n_classes = 2
height=68
width=68
directory='./pred_FCN/mask'+sys.argv[2];
#epoch=int(sys.argv[4])
epoch= 40
new = 1
#new=int(sys.argv[5])
print('************* seg_path : '+segs_path)
print('************* number of classes : '+str(n_classes))
print('************* directory : '+directory)
print('************* epochs : '+str(epoch))

if not os.path.exists(directory):
    print("creating directory ")
    os.makedirs(directory)
    train_flag=1;
    
else:
    print("directory already exists")
    train_flag=1;	
    
start=int(sys.argv[1]);
print('************ Cross fold : '+str(int((start+4)/4)))
actual_set=np.array([354,355,356,357,358,359,360,361,362,382,386,387,389,390,394,395])
actual_set=np.roll(actual_set,-1*start)
train_matrix=actual_set[0:8];
dev_matrix=actual_set[8:12];
test_matrix=actual_set[12:16];

print(actual_set)
print(train_matrix)
print(dev_matrix)
print(test_matrix)

X_train,Y_train,X_test,Y_test,X_dev,Y_dev,name,name_dev=img_utils.imageSegmentationGenerator(images_path , segs_path ,  batch_size,  n_classes , height , width, test_matrix,train_matrix , dev_matrix)


if(new==1):
	print('*********new model fitting*********')
	segnet_model=model.FCN8(n_classes ,68, 68)
	#segnet_model.summary()
	segnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	history=segnet_model.fit(X_train, Y_train,epochs=epoch,batch_size=16,validation_data=(X_test,Y_test),verbose=1)

#############################################################
# Save the model not the weights 
#############################################################
	#fcn_model.save_weights(directory+'/model_weight',overwrite=True)
	print('************ path : '+directory+'/model_cv'+str(int(start/4)))
	segnet_model.save(directory+'/model_cv'+str(int(start/4)))
else:
	print('*********old model fitting*********')
	print('************ path : '+directory+'/model_cv'+str(int(start/4)))
	segnet_model = load_model(directory+'/model_cv'+str(int(start/4)))
	scores = segnet_model.evaluate(X_test, Y_test, verbose=0)
	print(scores)
	print('**************** Score : '+str(scores)+' *****************')



#############################################################
# Predictions
#############################################################
test_path=directory+'/test_cv'+str(int(start/4))
dev_path=directory+'/dev_cv'+str(int(start/4))
if not os.path.exists(test_path):
    print("creating test directory : "+test_path)
    os.makedirs(test_path)
else:
    print("image directory already exists")

img_utils.predicted_output(segnet_model,X_test,name,test_path, height,width,n_classes)

if not os.path.exists(dev_path):
    print("creating development directory : "+dev_path)
    os.makedirs(dev_path)
else:
    print("image directory already exists")

img_utils.predicted_output(segnet_model,X_dev,name_dev,dev_path, height,width,n_classes)
