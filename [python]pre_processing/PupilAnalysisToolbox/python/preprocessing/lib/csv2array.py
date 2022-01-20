#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:03:35 2020

@author: yutasuzuki
"""

import numpy as np
from zeroInterp import zeroInterp
import matplotlib.pyplot as plt
from pre_processing import re_sampling

def csv2array(dat, eyes, fs, normalize = False,s_trg = 'Start_Experiment' ):
   
    events = {'MSG':[]}
    eyeData= {'Right':[],'Left':[]}

    # msg_type = ['SFIX','EFIX','SSACC','ESACC','SBLINK','EBLINK','MSG']
    
    start = False
    for line in dat:
        if start:
            if not line[5]=='NaN':
                eyeData['Left'].append([int(line[1]),
                                        float(line[5]),
                                        (line[9])])
            if not line[14]=='NaN':
                eyeData['Right'].append([int(line[1]),
                                         float(line[14]),
                                         (line[13])])
                
            events['MSG'].append([int(line[1]),line[3],line[4]])
        else:
            if s_trg in line:
                start = True
                                 
    pL = np.array([p[1] for p in eyeData['Left']])
    pR = np.array([p[1] for p in eyeData['Right']])
    
    # xL = np.array([p[1] for p in eyeData['Left']])
    # xR = np.array([p[1] for p in eyeData['Right']])
    
    # yL = np.array([p[2] for p in eyeData['Left']])
    # yR = np.array([p[2] for p in eyeData['Right']])
    
    timeStampL = np.array([p[0] for p in eyeData['Left']])
    timeStampR = np.array([p[0] for p in eyeData['Right']])
       
    initialTimeVal = timeStampR[0] if timeStampL[0] > timeStampR[0] else timeStampL[0]

    timeStampL = [round((t - initialTimeVal) * (fs / 10**6)) for t in timeStampL]
    timeStampR = [round((t - initialTimeVal) * (fs / 10**6)) for t in timeStampR]
    
    mmName = list(events.keys())
    for m in mmName:
        events[m] = [[int(((p[0]- initialTimeVal) * (fs / 10**6))),p[1],p[2]]
                          for p in events[m]]
    
    if eyes == 1: # both eyes
        timeLen = np.max(timeStampR) if np.max(timeStampR) > np.max(timeStampL) else np.max(timeStampL)
    
        pupilData = np.zeros((2,timeLen+1))
        pupilData[0,timeStampL] = pL
        pupilData[1,timeStampR] = pR
        
        # xData = np.zeros((2,timeLen+1))
        # xData[0,timeStampL] = xL
        # xData[1,timeStampR] = xR
        
        # yData = np.zeros((2,timeLen+1))
        # yData[0,timeStampL] = yL
        # yData[1,timeStampR] = yR
        dataLen = pupilData.shape[1]
   
        ind = np.argwhere(abs(np.diff(pupilData[0,])) > 1)
        pupilData[0,ind] = 0
        # xData[0,ind] = 0
        # yData[0,ind] = 0
        
        ind = np.argwhere(abs(np.diff(pupilData[1,])) > 1)
        pupilData[1,ind] = 0
        # xData[1,ind] = 0
        # yData[1,ind] = 0
     
    pupilData = zeroInterp(pupilData.copy(),fs,3)
    print('Interpolated array = ' + str(pupilData['interpolatedArray']) + 
          ' out of ' + str(pupilData['pupilData'].shape[1]))
    
    if pupilData['interpolatedArray'][0] < pupilData['interpolatedArray'][1]:
        pupilData = pupilData['pupilData'][0,:].reshape(1,pupilData['pupilData'].shape[1])
    else:
        pupilData = pupilData['pupilData'][1,:].reshape(1,pupilData['pupilData'].shape[1])
    
    pupilData = re_sampling(pupilData.copy(),dataLen)

    # plt.plot(np.diff(tmp[0,]))
    # plt.ylim([-50,50])
    # plt.xlim([10000, 15000])
    
    # plt.plot(tmp[0,])
    # plt.xlim([10000, 15000])
            
    pupilData = np.mean(pupilData,axis=0)
    # xData = np.mean(xData,axis=0)
    # yData = np.mean(yData,axis=0)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(pupilData.T,color="k")
    plt.ylim([2,8])
    
    plt.subplot(1,3,2)
    plt.plot(np.diff(pupilData).T,color="k")
    plt.ylim([-5,5])
    
    plt.subplot(1,3,3)
    plt.plot(pupilData.T,color="k")
    plt.xlim([10000, 11200])
    plt.ylim([2,8])
    
      
    if normalize:
        ave = np.mean(pupilData)
        sigma = np.std(pupilData)       
        pupilData = (pupilData - ave) / sigma
  
    return pupilData,events,initialTimeVal,int(fs)
