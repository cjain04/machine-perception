
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_report
import cv2
import numpy as np
import argparse
import os
from os.path import join
import glob


#get training path and store category names
train_path=os.path.abspath("<dataset path>")

training_names=os.listdir(train_path)
print (training_names)

#Processing Training set
image_paths=[]
image_classes=[]
rawimage_pix=[]
class_id=0
ds_names=len(training_names)
labels=[]
color_features=[]

def get_imgfiles(path):
	all_files=[]
	all_files.extend([join(path,fname)
			for fname in glob.glob(path+"/*")])
	#print all_files
	return all_files

for training_names,label in zip(training_names,range(ds_names)):
	class_path=join(train_path,training_names)
	class_files=get_imgfiles(class_path)
	image_paths+=class_files
	#print image_paths
	labels+=[class_id]*len(class_files)
	class_id+=1
	#print labels

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size(32 x 32), then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


def extract_color_histogram(image,bins=(8,8,8)):
	hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	hist=cv2.calcHist([hsv],[0,1,2],None,bins,[0 ,180, 0, 256, 0, 256])
	
	#inplace normalization for opencv 3
	cv2.normalize(hist,hist)
	#print hist.flatten().shape
	return hist.flatten()

for (i,image_path) in enumerate(image_paths):
	image=cv2.imread(image_path)
	#extract a color histogram from the image
	raw=image_to_feature_vector(image)
	rawimage_pix.append(raw)

	hist=extract_color_histogram(image)
	color_features.append(hist)

	#show update every 5 image
	if i>0 and i% 5==0:
	 	print("[INFO] processed {}/{}".format(i,len(image_paths)))

train_feat=color_features
train_labels=labels	 	

#Testing set
test_path=os.path.abspath("<dataset path>")
testing_names=os.listdir(test_path)
print (testing_names)

#Processing Training set
image_paths=[]
image_classes=[]
class_id=0
nbrnames=len(testing_names)
labels=[]
color_features=[]


for testing_names,label in zip(testing_names,range(ds_names)):
	class_path=join(test_path,testing_names)
	class_files=get_imgfiles(class_path)
	image_paths+=class_files
	#print image_paths
	labels+=[class_id]*len(class_files)
	class_id+=1

#loop over the input images
for (i,image_path) in enumerate(image_paths):
	image=cv2.imread(image_path)
	#extract a color histogram from the image
	hist=extract_color_histogram(image)
	color_features.append(hist)

	 #show update every 5 image
	if i>0 and i% 5==0:
	 	print("[INFO] processed {}/{}".format(i,len(image_paths)))

test_feat=color_features
test_labels=labels	 

print len(train_feat)
print len(train_labels)
print len(test_feat)
print len(test_labels)

train_feat=np.array(train_feat)
train_labels=np.array(train_labels)
test_feat=np.array(test_feat)
test_labels=np.array(test_labels)



#model=KNeighborsClassifier(3)
#Making Predictions
model=LinearSVC()
model.fit(train_feat,train_labels) 

predictions =model.predict(test_feat)
acc = model.score(test_feat, test_labels)
print (acc)
output=np.vstack((predictions,test_labels))
print (output)
#print accuracy_score(test_labels,predictions)
print (classification_report(test_labels,predictions))
