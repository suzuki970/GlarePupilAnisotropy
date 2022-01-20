
import numpy as np
from zeroInterp import zeroInterp
import matplotlib.pyplot as plt
# import math
from pre_processing import re_sampling

def asc2array(dat, eyes, mmFlag = False,normalize = False,s_trg = 'Start_Experiment' ):
   
    events = {'SFIX':[],'EFIX':[],'SSACC':[],'ESACC':[],'SBLINK':[],'EBLINK':[],'MSG':[]}
    eyeData= {'Right':[],'Left':[]}

    msg_type = ['SFIX','EFIX','SSACC','ESACC','SBLINK','EBLINK','MSG']
    
    start = False
    for line in dat:
        if start:
            if len(line) > 3:
                if line[0].isdecimal() and line[1].replace('.','',1).isdigit() :
                    eyeData['Left'].append([float(line[0]),
                                            float(line[1]),
                                            float(line[2]),
                                            float(line[3])])
                if line[0].isdecimal() and line[4].replace('.','',1).isdigit() :
                    eyeData['Right'].append([float(line[0]),
                                             float(line[4]),
                                             float(line[5]),
                                             float(line[6])])
                
            for m in msg_type:
                if line[0] == m:
                     events[m].append(line[1:])
        else:
            if 'RATE' in line:
                fs = float(line[5])
            if s_trg in line:
                start = True
                initialTimeVal = int(line[1])
                 
    pL = np.array([p[3] for p in eyeData['Left']])
    pR = np.array([p[3] for p in eyeData['Right']])
    
    xL = np.array([p[1] for p in eyeData['Left']])
    xR = np.array([p[1] for p in eyeData['Right']])
    
    yL = np.array([p[2] for p in eyeData['Left']])
    yR = np.array([p[2] for p in eyeData['Right']])
    
    timeStampL = np.array([p[0] for p in eyeData['Left']])
    timeStampR = np.array([p[0] for p in eyeData['Right']])
    
    # events_time = [int(t[0]- timeStampL[0]) for t in dat_msg]
    # events_msg = [t[1] for t in dat_msg]
    
    # initialTimeVal = timeStampR[0] if timeStampL[0] > timeStampR[0] else timeStampL[0]
      
    # events_response = [for e in events['MSG'] if e[1] == 'one_stream']

    timeStampL = [int((t - initialTimeVal)*(fs/1000)) for t in timeStampL]
    timeStampR = [int((t - initialTimeVal)*(fs/1000)) for t in timeStampR]
    
    # if eyes == 1: # both eyes
    timeLen = np.max(timeStampR) if np.max(timeStampR) > np.max(timeStampL) else np.max(timeStampL)

    pupilData = np.zeros((2,timeLen+1))
    pupilData[0,timeStampL] = pL
    pupilData[1,timeStampR] = pR
    
    xData = np.zeros((2,timeLen+1))
    xData[0,timeStampL] = xL
    xData[1,timeStampR] = xR
    
    yData = np.zeros((2,timeLen+1))
    yData[0,timeStampL] = yL
    yData[1,timeStampR] = yR
    
    dataLen = pupilData.shape[1]
    pupilData = re_sampling(pupilData.copy(),round(dataLen/4))

    for i in np.arange(2):
        ind = np.argwhere(abs(np.diff(pupilData[i,])) > 50)
        pupilData[i,ind] = 0  
    
    if normalize:
        tmp_p = abs(pupilData.copy())
        if mmFlag:
            tmp_p = 1.7*(10**(-4))*480*np.sqrt(tmp_p)
        
        ind_nonzero = np.argwhere(tmp_p[0,:] != 0).reshape(-1)
        ave_left = np.mean(tmp_p[0,ind_nonzero])
        sigma_left = np.std(tmp_p[0,ind_nonzero])       
        ind_nonzero = np.argwhere(tmp_p[1,:] != 0).reshape(-1)
        ave_right = np.mean(tmp_p[1,ind_nonzero])
        sigma_right = np.std(tmp_p[1,ind_nonzero])       
        ave = np.mean([ave_left,ave_right])
        sigma = np.mean([sigma_left,sigma_right])
        
    pupilData = zeroInterp(pupilData.copy(),fs/4,10)
    interplatedArray = pupilData['interpolatedArray'][0]
    print('Interpolated array = ' + str(pupilData['interpolatedArray']) + 
          ' out of ' + str(pupilData['pupilData'].shape[1]))
    
    if pupilData['interpolatedArray'][0] < pupilData['interpolatedArray'][1]:
        xData = xData[0,:].reshape(-1)
        yData = yData[0,:].reshape(-1)
    else:
        xData = xData[1,:].reshape(-1)
        yData = yData[1,:].reshape(-1)
       
    if eyes == 1: 
        if pupilData['interpolatedArray'][0] < pupilData['interpolatedArray'][1]:
            pupilData = pupilData['pupilData'][0,:].reshape(1,pupilData['pupilData'].shape[1])
            useEye = 'L'
        else:
            pupilData = pupilData['pupilData'][1,:].reshape(1,pupilData['pupilData'].shape[1])
            useEye = 'R'
    else: # both eyes
        pupilData = pupilData['pupilData']
        useEye = 'both'
        
    pupilData = re_sampling(pupilData.copy(),dataLen)
    
    ############ micro-saccade #########################
    dt = 1/fs
    v = np.zeros(len(xData))
    for i in np.arange(2,len(xData)-2):
        v[i] = (xData[i+2]+xData[i+1]-xData[i-2]-xData[i-1])
        # / (6*dt)
    
    for i in np.arange(2,len(xData)-2):
        if xData[i+2] * xData[i+1] * xData[i-2] * xData[i-1] == 0:
            v[i-50:i+50] = np.nan
    
    v[v>100] = np.nan
    v[xData>900] = np.nan
    v[xData<700] = np.nan
   
    # plt.figure()
    # plt.hist(v,bins=1000)
    # plt.xlim([-25, 25])
    # plt.xlim([0, 500])
    # sigma_m = np.nanmedian(v**2) - (np.nanmedian(v)**2)
    sigma_m = np.nanstd(v)
    ramda = 2
    upsilon = ramda*sigma_m
    # if upsilon > 10:
    #     upsilon=6
    # print('th = ' + str(upsilon))
    # print('std = ' + str(np.nanstd(v)))
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(v)
    # plt.plot((xData-np.mean(xData))*0.1-5)
    plt.hlines(upsilon, 0, len(xData), "red", linestyles='dashed')
    plt.xlim([200000, 205000])
    plt.ylim([-10, 20])
    # plt.ylim([-1000, 2000])
  
    ind = np.argwhere(abs(v) > upsilon)
    rejectNum = []
    for i in np.arange(len(ind)-1):
        if ind[i+1] == ind[i]+1:
            rejectNum.append(i+1)
    
    ind = np.delete(ind,rejectNum,axis=0)
    mSaccade = np.zeros(len(xData))
    mSaccade[ind] = v[ind]
            
    ############ data plot #########################
    pupilData = np.mean(pupilData,axis=0)
    # xData = np.mean(xData,axis=0)
    # yData = np.mean(yData,axis=0)
    
    
    # plt.plot(pupilData.T,color="k")
    # plt.ylim([0,10000])
    
    # plt.subplot(2,3,2)
    # plt.plot(np.diff(pupilData).T,color="k")
    # plt.ylim([-50,50])
    
    plt.subplot(1,3,2)
    plt.plot(pupilData.T)
    plt.xlim([200000, 210000])
    # plt.ylim([20000,10000])
    
    plt.subplot(1,3,3)
    plt.plot(np.diff(pupilData).T)
    plt.xlim([200000, 210000])
    plt.ylim([-50,50])
    # plt.subplot(2,3,4)
    # plt.plot(pupilData.T,color="k")
    # plt.xlim([500000, 550000])
    # plt.ylim([0,10000])
    
    # plt.subplot(2,3,5)
    # plt.plot(pupilData.T,color="k")
    # plt.xlim([1000000, 1050000])
    # plt.ylim([0,10000])
    
    # plt.subplot(2,3,6)
    # plt.plot(pupilData.T,color="k")
    # plt.xlim([2000000, 2050000])
    # plt.ylim([0,10000])
    
    if mmFlag:
         pupilData = abs(pupilData)
         pupilData = 1.7*(10**(-4))*480*np.sqrt(pupilData)
    
    if normalize:
        # ave = np.mean(pupilData)
        # sigma = np.std(pupilData)       
        pupilData = (pupilData - ave) / sigma
        
        # ind_fix = [int(int(e[0])-initialTimeVal) for e in events['MSG'] if e[1] == 'Start_Pesentation']
        # ind_fix.append(int(int(events['MSG'][-1][0])-initialTimeVal))
        # for i in np.arange(len(ind_fix)-1):
        #     # pupilData[ind_fix[i]:ind_fix[i+1]] = (pupilData[ind_fix[i]:ind_fix[i+1]] - np.mean(pupilData[ind_fix[i]:ind_fix[i+1]])) / np.std(pupilData[ind_fix[i]:ind_fix[i+1]])
        #     pupilData[ind_fix[i]:ind_fix[i+1]] = pupilData[ind_fix[i]:ind_fix[i+1]] - np.mean(pupilData[ind_fix[i]+0:(ind_fix[i]+1000)])
        #     # pupilData[ind_fix[i]:ind_fix[i+1]] = (pupilData[ind_fix[i]:ind_fix[i+1]] - np.mean(pupilData[ind_fix[i]:ind_fix[i]+fs])) / np.std(pupilData[ind_fix[i]:ind_fix[i]+fs])
    eyeData = {'pupilData':pupilData,
               'gazeX':xData,
               'gazeY':yData,
               'mSaccade':mSaccade,
               'useEye':useEye
               }
    return eyeData,events,initialTimeVal,int(fs)
