#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:25:30 2020

@author: yuta
"""

import scipy.io
import numpy as np
import json
import os

tmp_e3 = scipy.io.loadmat('/Users/yuta/Desktop/Github/Experimental_data/P04/figure/data_202003_6.mat')
keyNum = list(tmp_e3.keys())
for i in np.arange(3):
    tmp_e3.pop(keyNum[i])
    
keyNum = list(tmp_e3.keys())
for ikey in keyNum:
    if tmp_e3[ikey].shape[1] == 1:
        tmp_e3[ikey] = tmp_e3[ikey].reshape(-1).tolist()
    else:
        tmp_e3[ikey] = tmp_e3[ikey].tolist()
        
with open(os.path.join("/Users/yuta/Desktop/Github/Experimental_data/P04/figure/data_e3.json"),"w") as f:
        json.dump(tmp_e3,f)
   
#####################
tmp_e2 = scipy.io.loadmat('/Users/yuta/Desktop/Github/Experimental_data/P04/figure/data_exp2.mat')
keyNum = list(tmp_e2.keys())
for i in np.arange(3):
    tmp_e2.pop(keyNum[i])
    
keyNum = list(tmp_e2.keys())
for ikey in keyNum:
    if tmp_e2[ikey].shape[1] == 1:
        tmp_e2[ikey] = tmp_e2[ikey].reshape(-1).tolist()
    else:
        tmp_e2[ikey] = tmp_e2[ikey].tolist()
        
with open(os.path.join("/Users/yuta/Desktop/Github/Experimental_data/P04/figure/data_e2.json"),"w") as f:
        json.dump(tmp_e2,f)
       
#####################
tmp_disp = scipy.io.loadmat('/Users/yuta/Desktop/Github/Experimental_data/P04/figure/lut_desplay++.mat')
keyNum = list(tmp_disp.keys())
for i in np.arange(3):
    tmp_disp.pop(keyNum[i])
    
keyNum = list(tmp_disp.keys())
for ikey in keyNum:
    if tmp_disp[ikey].shape[1] == 1:
        tmp_disp[ikey] = tmp_disp[ikey].reshape(-1).tolist()
    else:
        tmp_disp[ikey] = tmp_disp[ikey].tolist()
        
with open(os.path.join("/Users/yuta/Desktop/Github/Experimental_data/P04/figure/data_disp.json"),"w") as f:
        json.dump(tmp_disp,f)
       
#####################
tmp_e1 = scipy.io.loadmat('/Users/yuta/Desktop/Github/Experimental_data/P04/figure/data_pupil.mat')
keyNum = list(tmp_e1.keys())
for i in np.arange(3):
    tmp_e1.pop(keyNum[i])
    
keyNum = list(tmp_e1.keys())
for ikey in keyNum:
    if tmp_e1[ikey].shape[1] == 1:
        tmp_e1[ikey] = tmp_e1[ikey].reshape(-1).tolist()
    else:
        tmp_e1[ikey] = tmp_e1[ikey].tolist()
        
with open(os.path.join("/Users/yuta/Desktop/Github/Experimental_data/P04/figure/data_e1.json"),"w") as f:
        json.dump(tmp_e1,f)
