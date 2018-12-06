import random;
import glob;
import cv2
import numpy as np
import os
import scipy.io as sio

path = '../data/input/';

# subract the average value of image 

def RandPick(subject):
	random.seed(45)
	#print(path + subject + '*.*')
	total = glob.glob(path + subject + '*.*');
	n = len(total);
	#print(n)
	arr = [i for i in range(n)];
	random.shuffle(arr);
	#arr = arr[0:num];
	train = [total[arr[i]] for i in range(500)];
	test = [total[arr[i]] for i in range(500,850)];
	dev = [total[arr[i]] for i in range(850,950)];
	return train,test,dev

def getImageArr(path):
	#new = np.zeros((128,128,3),dtype = "float64")
	img = cv2.imread(path, 1)
	#print('image_read')

	img=img.astype('float64')
	#img[:, :, 0] -= 103.939
	#img[:, :, 1] -= 116.779
	#img[:, :, 2] -= 123.68
	#new[0:68,0:68,:] = img
	#if imgNorm == 'sub':

	#    img=img-img_mean
	#else:
	img=img/255
	return img
    
def getSegmentationArr( path , nClasses = 2 ,  width = 68 , height =68 ):
	seg_labels = np.zeros((  width , height  , nClasses ))
	img = cv2.imread(path, 1)
	img = img[:, : , 0]
	#new = np.zeros((128,128),dtype = "int")
	#new[0:68,0:68] = img
	for c in range(nClasses):
		seg_labels[: , : , c ] = (img == c ).astype(int)
	#print(seg_labels.shape)
	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels
    

def readImage(subject, mask):
	path = '../data/input/';
	# number of test image
	
	Xtrain_dict=dict()
	Ytrain_dict=dict()
	Xtest = []
	Ytest = []
	Xdev = []
	Ydev = []
	out_list = []
	label_path = '../data/output/mask'+str(mask)+'/'
	print('label_path'+label_path)
	
	for sub in subject:
		print("Subject: "+str(sub))
		train_list,test_list,dev_list = RandPick(sub)
		Xtrain = []
		Ytrain = []
	
		for i in train_list:
			
			Xtrain.append(getImageArr(i))
			test_image = i.split("/")[-1][0:-4]
			test_str = label_path+test_image+".png"
			Ytrain.append(getSegmentationArr(test_str))

		Xtrain_dict[sub] = np.array(Xtrain)
		Ytrain_dict[sub] = np.array(Ytrain)

		for i in test_list:
			Xtest.append(getImageArr(i))
			test_image = i.split("/")[-1][0:-4]
			test_str = label_path+test_image+".png"
			Ytest.append(getSegmentationArr(test_str))
			out_list.append(test_image)
		for i in dev_list:
			Xdev.append(getImageArr(i))
			test_image = i.split("/")[-1][0:-4]
			test_str = label_path+test_image+".png"
			Ydev.append(getSegmentationArr(test_str))
	
	print(Xtrain_dict.keys())
	print(Ytrain_dict.keys())
	print(len(Xtest))
	print(len(Ytest))
	print(len(Xdev))
	print(len(Ydev))
	print('outlist :'+str(len(out_list)))

	return Xtrain_dict,Ytrain_dict,np.array(Xtest),np.array(Ytest),np.array(Xdev),np.array(Ydev),out_list

def predicted_output(model,X_test,name,dir_path, height=68,width=68,n_classes=2):
	batch_len=X_test.shape[0]
	pr = model.predict(X_test)
	print('images output')
	pr_1 = np.reshape(pr,(batch_len,height,width,n_classes)).argmax( axis=3 )
	pr_2 = np.reshape(pr,(batch_len,height,width,n_classes))
	print(pr.shape)
	for num in range(batch_len):
		# seg_img = np.zeros((height , width , 3))
		out_name=dir_path+'/'+name[num]+'.png'
		cv2.imwrite(out_name , pr_1[num]*80)
		sio.savemat(dir_path+'_data.mat', {'y_pred':pr_1,'name_list':name,'y_pred_map':pr_2})

def readImage_sub(subject,mask):
	f1_videos = dict()
	f1_label = dict()
	f1_list = dict()
	F1 =sorted(glob.glob('../data/input/F1*.jpg'))
	label_path = '../data/output/mask'+str(mask)+'/'
	for i in F1:
		vid_num = i.split('_')[-2]
		sample = i.split('/')[-1][0:-4]
		file_name = label_path+sample+'.png'
		img = getImageArr(i)
		label = getSegmentationArr(file_name)
		if(vid_num in f1_videos):
			f1_videos[vid_num].append(img)
			f1_label[vid_num].append(label)
			f1_list[vid_num].append(sample)
		else:
			f1_videos[vid_num] = []
			f1_videos[vid_num].append(img)
			f1_label[vid_num] = []
			f1_label[vid_num].append(label)
			f1_list[vid_num] = []
			f1_list[vid_num].append(sample)
	return f1_videos,f1_label,f1_list
	
#readImage(subject, mask,num)
