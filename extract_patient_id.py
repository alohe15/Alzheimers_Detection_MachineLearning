#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:08:04 2019

@author: arnavlohe
"""

import pandas as pd
df = pd.read_csv('/Users/arnavlohe/Desktop/nc.csv')
#print(df)

sample = '/home/ec2-user/MCIc_Segmented/duplicateADNI_127_S_1427_MR_MP-RAGE__br_raw_20100820102300886_155_S90938_I191032_be_pveseg.nii'

def ID(string):
    lst = string.split('/')
    string = lst[4]
    lst = string.split('_')
    #return(lst)
    id = lst[1] + '_' + lst[2] + '_' + lst[3]
    return(id)

#print(ID(sample))

def imageno(string):
    lst = string.split('/')
    string = lst[4]
    lst = string.split('_')
    lst = lst[1:]
    for item in lst:
        if 'I' in item:
            return(item)

#print(imageno(sample))

#functions return patient ID number and image number, respectively

#for i in range(0, 768):
 #   df['ID'][i] = ID(df['ID'][i])

#print(df['ID'])

series = pd.Series(index = range(0, 1512))
#print(series)

for i in range(0, 1512):
    series[i] = imageno(df['ID'][i])
    

#print(series)
"""for i in range(0, 768):
    try:
        series[i] = imageno(df['ID'][i])
    except:
        pass"""

df['ImageNo'] = series

for i in range(0, 1512):
    df['ID'][i] = ID(df['ID'][i])

print(df)

c_csv = df.to_csv('/Users/arnavlohe/Desktop/nc2.csv', index = None, header=True)