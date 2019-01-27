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

    #lists to store voxels of grey, white, CSF
    one = []
    two = []
    three = []

    #counting grey, white, and CSF voxels
    for i in range(0,dim1):
        for j in range(0, dim2):
            for k in range(0, dim3):
                w = data[i,j,k]
                if (w == 1.0):
                    one.append(w)
                if (w == 2.0):
                    two.append(w)
                if (w == 3.0):
                    three.append(w)
                print('.')
    print("COUNTED")

    #returning tuple
    return((path, len(one), len(two), len(three)))

#lists to store the tuples
c_list = []
nc_list = []
c_df = pd.DataFrame(columns=["ID", "CSF", "GREY", "WHITE"])
nc_df = pd.DataFrame(columns=["ID", "CSF", "GREY", "WHITE"])
#print(c_df)
#print(nc_df)


#iterating through the directories
for file in os.listdir(c_dir):
    if is_relevant(file):
        print("c ye")
        c_df["ID"] = quantify(c_dir + '/' + file)[0]
        c_df["CSF"] = quantify(c_dir + '/' + file)[1]
        c_df["GREY"] = quantify(c_dir + '/' + file)[2]
        c_df["WHITE"] = quantify(c_dir + '/' + file)[3]
        c_list.append(quantify(c_dir + '/' + file))

for file in os.listdir(nc_dir):
    if is_relevant(file):
        print("nc ye")
        nc_df["ID"] = quantify(nc_dir + '/' + file)[0]
        nc_df["CSF"] = quantify(nc_dir + '/' + file)[1]
        nc_df["GREY"] = quantify(nc_dir + '/' + file)[2]
        nc_df["WHITE"] = quantify(nc_dir + '/' + file)[3]
        nc_list.append(quantify(nc_dir +  '/' + file))

c_csv = c_df.to_csv ('/home/ec2-user/c.csv', index = None, header=True)
nc_csv = nc_df.to_csv ('/home/ec2-user/nc.csv', index = None, header=True)
