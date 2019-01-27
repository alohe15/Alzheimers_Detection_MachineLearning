#script to retrieve volume quantifications for grey matter, white matter, and CSF

#libraries
import nibabel as nib
import numpy as np
import os
import pandas as pd
import numba
from numba import jit

#c is one class of patient, nc is another
c_dir = '/home/ec2-user/MCIc_Segmented'
nc_dir = '/home/ec2-user/MCInc_Segmented'

#determining whether a given file will be used in volume quantification`
def is_relevant(path):
    g_w = 'pveseg' in path
    unzipped = '.gz' not in path
    if g_w and unzipped:
        return True

#function to return the volumes of respective tissue types as a (path, CSF, grey, white) tuple
@jit
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

#lists to store the tuples
c_list = []
nc_list = []
c_df = pd.DataFrame(columns=["ID", "CSF", "GREY", "WHITE"], index = range(0, 768))
nc_df = pd.DataFrame(columns=["ID", "CSF", "GREY", "WHITE"], index = range(0, 1512))
#print(c_df)
#print(nc_df)


#iterating through the directories
index = 0
c_count = 0
for file in os.listdir(c_dir):
    if is_relevant(file):
        print("c exists")
        tup = quantify(c_dir + '/' + file)
        print(tup[0], tup[1], tup[2], tup[3])
        nc_df["ID"][c_count] = tup[0]
        print("inserted")
        nc_df["CSF"][c_count] = tup[1]
        print("inserted")
        nc_df["GREY"][c_count] = tup[2]
        print("inserted")
        nc_df["WHITE"][c_count] = tup[3]
        print("inserted")
        print(str(c_count) + " c mages processed")
        c_count += 1

nc_count = 0
for file in os.listdir(nc_dir):
    if is_relevant(file):
        print("nc exists")
        tup = quantify(nc_dir + '/' + file)
        print(tup[0], tup[1], tup[2])
        nc_df["ID"][nc_count] = tup[0]
        print("inserted")
        nc_df["CSF"][nc_count] = tup[1]
        print("inserted")
        nc_df["GREY"][nc_count] = tup[2]
        print("inserted")
        nc_df["WHITE"][nc_count] = tup[3]
        print("inserted")
        print(str(nc_count) + " nc images processed")
        nc_count += 1

c_csv = c_df.to_csv('/home/ec2-user/c.csv', index = None, header=True)
nc_csv = nc_df.to_csv('/home/ec2-user/nc.csv', index = None, header=True)
