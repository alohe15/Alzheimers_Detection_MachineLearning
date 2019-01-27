#the actual program

import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import pandas as pd
import random
import os
import gzip

#c is one class of patient, nc is another
c_dir = '/home/ec2-user/MCIc_Segmented'
nc_dir = '/home/ec2-user/MCInc_Segmented'

c_patients = os.listdir(c_dir)
nc_patients = os.listdir(nc_dir)

c_list = []
nc_list = []

#function to filter whether an image is to be trained on
def is_relevant(path):
    g_w = 'pve_0' in path
    unzipped = '.gz' not in path
    if g_w and unzipped:
        return True

#appending images to lists
for file in os.listdir(c_dir):
    if is_relevant(file):
        c_list.append(nib.load(c_dir + '/' + file))
for file in os.listdir(nc_dir):
    if is_relevant(file):
        nc_list.append(nib.load(nc_dir +  '/' + file))

#creating a datfaframe with all the images and their classifications, 0 corresponds to 'nc' and 1 corresponds to 'c'
names = ['Images', 'Classification']
data = c_list + nc_list
np.random.shuffle(data)
classifications = list(np.zeros(len(c_list), dtype = int)) + list(np.ones(len(nc_list), dtype = int))
df = pd.DataFrame()
df["Images"] = data
df["Classification"] = classifications
#print(df)
df.to_csv('patients.csv', encoding = 'utf-8', index = False)
df2 = pd.read_csv('patients.csv')
print(df2)

"""
x = tf.placeholder(tf.float32, shape=[None, 256,256,166,1], name='x')
y = tf.placeholder(tf.float32, shape=[None, labels], name='y')"""
