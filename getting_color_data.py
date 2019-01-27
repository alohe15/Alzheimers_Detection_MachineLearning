#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 21:31:41 2019

@author: arnavlohe
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:00:08 2018

@author: arnavlohe
"""

import nibabel as nib
import numpy as np
import os
import pandas as pd

path3 = '/Users/arnavlohe/Desktop/ADNI_002_S_0729_MR_MP-RAGE_REPEAT_br_raw_20080409214821885_1_S48530_I101815_be_be_pveseg.nii'
img3 = nib.load(path3)
data3 = img3.get_fdata()


#CSF
print(data3[83,143,157]) #1.0
#Grey Matter
print(data3[83,171,164]) #2.0
#White Matter
print(data3[83,107,169]) #3.0

#This means I can only analyze the pveseg images



    