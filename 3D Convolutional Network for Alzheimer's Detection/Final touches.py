import os
import pandas as pd
import numpy as np
import skimage
from skimage import transform
import nibabel

ssdata_dir='path_to_skull_stripped_images'
aldata_dir='path_to_images_of_patients_with_alzheimers'
nldata_dir='path_to_images_of_normal_control'
mcidata_dir='path_to_images_of_patients_with_mild_cognitive_impairment'
labels_df=pd.read_csv('path_to_dataset_datasheet',index_col=2)                          #indexed by the patient id

#labeling and object formation

for file in os.listdir(ssdata_dir):
    netdata=[]                                              #will be used for numpy object
    try:
        img = nibabel.load(os.path.join(ssdata_dir, file))  # loading the image
        img = img.get_data()                           # accessing image array
        img = skimage.transform.resize(img, (106, 106, 120))#resizing the image to dimensions(106,106,120)
        id = file.partition('.')                            #accessing patient id numbers
        id = id[0].partition('_2')[0]
        label=labels_df.get_value(id[0], 'Screen.Diagnosis') #getting the label
        if np.unique(label == 'NL'):
            labelar = np.array([1, 0, 0])
            netdata.append([img, labelar])                          #one hot encoding and saving numpy object
            np.save(os.path.join(nldata_dir, id[0] + id[1]), netdata)
        elif np.unique(label == 'AD'):
            labelar = np.array([0, 1, 0])
            netdata.append([img, labelar])
            np.save(os.path.join(aldata_dir, id[0] + id[1]), netdata)
        elif np.unique(label == 'MCI'):
            labelar = np.array([0, 0, 1])
            netdata.append([img, labelar])
            np.save(os.path.join(mcidata_dir, id[0] + id[1]), netdata)
    except:
        continue

#normalisation

totalnum=[]         #total number of pixels in the image
mean=[]             #mean of the pixels in the image
nummax=[]           #maximum value of pixels in the image
for file in os.listdir(aldata_dir):
    img = np.load(os.path.join(aldata_dir,file))
    mean.append(np.mean(img[0][0]))
    totalnum.append((img[0][0].shape[0]*img[0][0].shape[1]*img[0][0].shape[2]))
    nummax.append(np.max(img[0][0]))
for file in os.listdir(nldata_dir):
    img = np.load(os.path.join(nldata_dir, file))
    mean.append(np.mean(img[0][0]))
    totalnum.append((img[0][0].shape[0]*img[0][0].shape[1]*img[0][0].shape[2]))
    nummax.append(np.max(img[0][0]))
for file in os.listdir(mcidata_dir):
    img = np.load(os.path.join(mcidata_dir, file))
    mean.append(np.mean(img[0][0]))
    totalnum.append((img[0][0].shape[0]*img[0][0].shape[1]*img[0][0].shape[2]))
    nummax.append(np.max(img[0][0]))
nummean=np.vdot(mean,totalnum)/np.sum(totalnum)           #mean value for the full dataset
nummax=np.max(nummax)                                     #max value for the full dataset

for file in os.listdir(aldata_dir):
    img = np.load(os.path.join(aldata_dir,file))
    img[0][0]=(img[0][0]-nummean)/nummax                 #normalisation(x-mean/max value)
    np.save(os.path.join(aldata_dir,file),img)
for file in os.listdir(nldata_dir):
    img = np.load(os.path.join(nldata_dir, file))
    img[0][0] =(img[0][0] - nummean) / nummax
    np.save(os.path.join(nldata_dir,file),img)
for file in os.listdir(mcidata_dir):
    img = np.load(os.path.join(mcidata_dir, file))
    img[0][0] =(img[0][0] - nummean) / nummax
    np.save(os.path.join(mcidata_dir, file),img)


