
import cv2
import glob
import itertools
import numpy as np
import os
import sys
import scipy.io as sio
#from matplotlib.pyplot import imread


def getImageArr( path , width , height ,  imgNorm = 'None'):

    img = cv2.imread(path, 1)
    #print('image_read')
    img=img.astype('float64')
    #if imgNorm == 'sub':
        
    #    img=img-img_mean
    #else:
    img=img/255
    return img
    

def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))

    img = cv2.imread(path, 1)
    #print('image read'+path)
    img = cv2.resize(img, ( width , height ))
    #print(img.shape)
    img = img[:, : , 0]

    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)

    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels


def imageSegmentationGenerator( images_path , segs_path , batch_size,  n_classes , height , width, test_matrix , train_matrix, dev_matrix):

	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'

	images = glob.glob( images_path + "*.jpg"  ) 
	images.sort()
	print(len(images))
	segmentations  = glob.glob( segs_path + "*.png"  ) 
	segmentations.sort()
	print(len(segmentations))
	#print(images[1:10])
	#print(segmentations[1:10])
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	X_dev = []
	Y_dev = []
	name_list=[]
	name_list_dev=[]
	for num in range(batch_size) :
		if(images[num].split('/')[-1].split(".")[0]==segmentations[num].split('/')[-1].split(".")[0]):
			#print(images[num].split('/')[-1].split(".")[0]+'  '+segmentations[num].split('/')[-1].split(".")[0])
			#print(segmentations[num].split('/')[-1].split(".")[0])
			v_num=int(segmentations[num].split('/')[-1].split(".")[0][3:6])
			# test data
			if(np.sum([test_matrix==v_num])):
				X_test.append( getImageArr(images[num] , width , height))
				Y_test.append( getSegmentationArr( segmentations[num] , n_classes , width , height )  )
				name_list.append(segmentations[num].split('/')[-1].split(".")[0])
			# train data
			elif(np.sum([train_matrix==v_num])):
				X_train.append( getImageArr(images[num] , width , height))
				Y_train.append( getSegmentationArr( segmentations[num] , n_classes , width , height )  )
			elif(np.sum([dev_matrix==v_num])):
				X_dev.append( getImageArr(images[num] , width , height))
				Y_dev.append( getSegmentationArr( segmentations[num] , n_classes , width , height )  )
				name_list_dev.append(segmentations[num].split('/')[-1].split(".")[0])
			else:
				c=0;

	print(len(X_train))
	print(len(Y_train))
	print(len(name_list))
	print(len(X_test))
	print(len(Y_test))
	print(len(X_dev))
	print(len(Y_dev))
	print(len(name_list_dev))
	return np.array(X_train) , np.array(Y_train), np.array(X_test) , np.array(Y_test), np.array(X_dev) , np.array(Y_dev), name_list, name_list_dev


def predicted_output(model,X_test,name,dir_path, height=68,width=68,n_classes=2):
    	batch_len=X_test.shape[0]
    	pr = model.predict(X_test)
	print('images output')
    	pr_1 = np.reshape(pr,(batch_len,height,width,n_classes)).argmax( axis=3 )
    	pr_2 = np.reshape(pr,(batch_len,height,width,n_classes))
    	print(pr.shape)
    	for num in range(batch_len):
        	#seg_img = np.zeros((height , width , 3))
        	out_name=dir_path+'/'+name[num]+'.png'
        	cv2.imwrite(out_name , pr_1[num]*80)
	sio.savemat(dir_path+'_data.mat', {'y_pred':pr_1,'name_list':name,'y_pred_map':pr_2})

