#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:16:55 2019

@author: arnavlohe
"""

import pandas as pd
import nibabel as nib
import numpy as np

path = '/Users/arnavlohe/Desktop/ADNI_002_S_0729_MR_MP-RAGE_REPEAT_br_raw_20070225105857428_72_S27091_I41585_be_be_pveseg.nii'
img = nib.load(path)
image = img.get_fdata()
#df = pd.DataFrame(columns = ['One', 'Two'], index = range(0, 5))
#print(df)


#values = [4, 5]
#df['One'][0] = 4
#print(df)

def quantify(path):
    #loading image
    img = nib.load(path)
    data = img.get_fdata()

    #extracting width, height, depth of 3D image
    dim1 = img.header.get_data_shape()[0]
    dim2 = img.header.get_data_shape()[1]
    dim3 = img.header.get_data_shape()[2]

    unique, counts = np.unique(data, return_counts=True)
    vals =  dict(zip(unique, counts))
    return ((path, vals[1], vals[2], vals[3]))


df = pd.DataFrame(columns=["ID", "CSF", "GREY", "WHITE"], index = range(0, 3))
print(df)
print(quantify(path))

"""
tup = ("idnumber", 2000, 3000, 4000)
print(tup[0], tup[1], tup[2], tup[3])
df["ID"][0] = tup[0]
print("inserted")
df["CSF"][0] = tup[1]
print("inserted")
df["GREY"][0] = tup[2]
print("inserted")
df["WHITE"][0] = tup[3]
print("inserted")
print(df)

prac_csv = df.to_csv('/Users/arnavlohe/Desktop/prac.csv', index = None, header=True)
"""