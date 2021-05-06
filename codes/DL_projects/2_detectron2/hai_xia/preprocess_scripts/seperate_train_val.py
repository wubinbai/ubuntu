import numpy as np
import glob
import numpy
import os
import shutil

TRAIN_VAL_RATIO = 4
image_files = glob.glob('b02/*jpg')
np.random.shuffle(image_files)
os.makedirs('train/xml',exist_ok=True)
os.makedirs('train/jpg',exist_ok=True)
os.makedirs('val/xml',exist_ok=True)
os.makedirs('val/jpg',exist_ok=True)
num_train = len(image_files) // (TRAIN_VAL_RATIO + 1)
for i in range(num_train*TRAIN_VAL_RATIO):
    #print(i)
    image_src = image_files[i]
    image_dst = 'train/jpg/' + (image_files[i].split('/')[-1])
    #print(image_src,image_dst)
    shutil.copy(image_src,image_dst)
    xml_src = 'xml/' + (image_files[i].split('/')[-1]).split('.')[0] + '.xml'
    xml_dst = 'train/xml/' + (image_files[i].split('/')[-1]).split('.')[0] + '.xml'
    #print(xml_src,xml_dst)
    shutil.copy(xml_src,xml_dst)


for i in range(num_train*TRAIN_VAL_RATIO,len(image_files)):
    #print(i)
    image_src = image_files[i]
    image_dst = 'val/jpg/' + (image_files[i].split('/')[-1])
    #print(image_src,image_dst)
    shutil.copy(image_src,image_dst)
    xml_src = 'xml/' + (image_files[i].split('/')[-1]).split('.')[0] + '.xml'
    xml_dst = 'val/xml/' + (image_files[i].split('/')[-1]).split('.')[0] + '.xml'
    #print(xml_src,xml_dst)
    shutil.copy(xml_src,xml_dst)

