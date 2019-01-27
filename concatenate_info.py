#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 11:00:20 2019

@author: arnavlohe
"""

#combining datasets into one consolidated csv with ALL patient and image information
import pandas as pd
import numpy as np

brain_df = pd.read_csv('/Users/arnavlohe/Desktop/newcombined.csv')
patient_df = pd.read_csv('/Users/arnavlohe/Desktop/combined_patientinfo.csv')
common = pd.read_csv('/Users/arnavlohe/Desktop/commonIDs.csv')
#print(common['Common'])

#Extracting Common Image IDs to both datasets
common_series = pd.Series(common['Common'])
common_series.dropna(inplace = True)
common_series = common_series.astype(int)
#print(common_series)

#Getting rid of junk rows and converting datatype of image ID to int
brain_df = brain_df[brain_df.ImageNo != 'AGITTAL']
brain_df['ImageNo'] = brain_df['ImageNo'].astype(int)
#print(brain_df)
#print(brain_df) #2279

#Converting datatype of image ID to int
patient_df['Image Data ID'] = patient_df['Image Data ID'].astype(int)
patient_df = patient_df.drop(columns = ['Modality', 'Description', 'Format', 'Downloaded', 'Type', 'Acq Date' ])
#print(patient_df) #2774

#creating empty columns to store patient age and sex
brain_df['Age'] = np.nan
brain_df['Sex'] = np.nan

#creating ... something
agesex_list = [[row[2], row[5], row[6]] for row in patient_df.itertuples()]
print(len(data_list)) #2775

j = 0
i = 0
k = 0

#creating ... another something 
data_list = [[row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]] for row in brain_df.itertuples()]
#print(data_list)
#print(len(data_list)) #2270

for i in range(0, 2270):
    for j in range(0, 2775):
        if (data_list[i])[1] == (agesex_list[j])[0]:
            (data_list[i]).append((agesex_list[j])[1])
            (data_list[i]).append((agesex_list[j])[2])
            print(str((data_list[i])[1]) + " added")

print(data_list)

final_df = pd.DataFrame(columns = ['Class', 'ImageNo', 'SubjectID', 'CSF', 'GREY', 'WHITE', 'TOTAL', 'G/T', 'W/T', 'Sex', 'Age'], index = range(0, 2270))
#print(final_df)

for i in range(0, 2270):
    final_df['Class'][i] = (data_list[i])[0]
    final_df['ImageNo'][i] = (data_list[i])[1]
    final_df['SubjectID'][i] = (data_list[i])[2]
    final_df['CSF'][i] = (data_list[i])[3]
    final_df['GREY'][i] = (data_list[i])[4]
    final_df['WHITE'][i] = (data_list[i])[5]
    final_df['TOTAL'][i] = (data_list[i])[6]
    final_df['G/T'][i] = (data_list[i])[7]
    final_df['W/T'][i] = (data_list[i])[8]
    final_df['Sex'][i] = (data_list[i])[9]
    final_df['Age'][i] = (data_list[i])[10]

print(final_df)

final_dataset = final_df.to_csv('/Users/arnavlohe/Desktop/final_dataset.csv')

    









