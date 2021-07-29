
import numpy as np
import numpy.matlib
from band_pass_filter import butter_bandpass_filter,lowpass_filter
import pandas
from scipy import interpolate

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def re_sampling(dat,num):
    re_sampled = []
    for d in dat:
        t = np.array(d)
        numX = np.arange(len(t))
        yy = interpolate.PchipInterpolator(numX, t)
        t_resample = np.linspace(0, len(t), num)
        re_sampled.append(yy(t_resample))
    
    return np.array(re_sampled)

def getNearestValue(in_y, num):
    idx = np.abs(np.asarray(in_y) - num).argmin()
    return idx

def moving_avg(in_y,windowL):
    
    if in_y.ndim == 1:
        in_y = in_y.reshape(1,len(in_y))
        
    tmp_y = in_y.copy()
    
    for trials in np.arange(len(in_y)):
        s = pandas.Series(in_y[trials,])
        in_y[trials,] = s.rolling(window=windowL).mean()
    
    out_y = []
    for trials in np.arange(len(in_y)):
        out_y.append(np.r_[tmp_y[trials,np.arange(windowL)],
                     in_y[trials,np.arange(windowL,in_y.shape[1])]])
    
    return np.array(out_y)

def reject_trials(y,thres,baselineData):
  
    ## reject trials when the velocity of pupil change is larger than threshold
    rejectNum=[]
    fx = np.diff(y, n=1)
    for trials in np.arange(y.shape[0]):
        ind = np.argwhere(abs(fx[trials,np.arange(baselineData[0],baselineData[2])]) > thres)
        # ind = np.argwhere(abs(fx[trials,np.arange(50,baselineData[2])]) > thres)
        
        if len(ind) > 0:
            rejectNum.append(trials)
            continue
            
        # if sum(np.isnan(y[trials,np.arange(baselineData[0],baselineData[2])])) > 0:
        #     rejectNum.append(trials)
        #     continue
          ## reject trials when number of 0 > 50#
        if sum(np.argwhere(y[trials,np.arange(baselineData[0],baselineData[2])] == 0)) > y.shape[0] / 2:
            rejectNum.append(trials)
            continue
        
    ## reject trials when the NAN includes
    tmp = np.argwhere(np.isnan(y) == True) 
    for i in np.arange(tmp.shape[0]):
        rejectNum.append(tmp[i,0])
        
    rejectNum = np.unique(rejectNum)
    set(rejectNum)
    return rejectNum.tolist()

    
# def pre_processing(y,fs,thres,windowL,timeLen,method,filt):
def pre_processing(dat,cfg):
    
    filt         = cfg['WID_FILTER']
    fs           = cfg['SAMPLING_RATE']
    windowL      = cfg['windowL']
    TIME_START   = cfg['TIME_START']
    TIME_END     = cfg['TIME_END']
    wid_base     = cfg['WID_BASELINE']
    method       = cfg['METHOD']
    thres        = cfg['THRES_DIFF']
    wid_analysis = cfg['WID_ANALYSIS']
    
    if isinstance(dat, list):
        rejectNum=[]
        out_y = []
        for i,p in enumerate(dat):
            
            timeWin = len(p)
            
            ## Smoothing
            s = pandas.Series(p)
            y = s.rolling(window=windowL).mean()
            y = np.array(y)
            timeWin = len(y)
            
            x = np.linspace(TIME_START[i],TIME_END[i],timeWin)
           
            # filtering
            # if len(filt) > 0:
            #     ave = np.nanmean(y)
            #     y = y - ave
            #     y = butter_bandpass_filter(y, filt[0], filt[1], fs, order=4)
            #     y = y + ave
        
            baselineData = np.array([getNearestValue(x,wid_base[0]),getNearestValue(x,wid_base[1]),getNearestValue(x,wid_analysis[i])])
            baselinePLR = y[np.arange(baselineData[0],baselineData[1])]
            baselinePLR_std = np.std(baselinePLR)
            # baselinePLR_std = np.tile(baselinePLR_std, (1,timeWin)).reshape(1,timeWin).T
            baselinePLR = np.mean(baselinePLR)
            # baselinePLR = np.tile(baselinePLR, (1,timeWin)).reshape(timeWin,dat.shape[0]).T   
           
            if method == 1:
                y = y - baselinePLR
            elif method == 2:
                y = (y - baselinePLR) / baselinePLR_std
            else:
                y = y 
                
            fx = np.diff(y)
            ind = np.argwhere(abs(fx[np.arange(baselineData[0],baselineData[2])]) > thres)
            # ind = np.argwhere(abs(fx) > thres)
        
            if len(ind) > 0:
                rejectNum.append(i)
                # continue
            
            ## reject trials when number of 0 > 50#
            if sum(np.argwhere(y[np.arange(baselineData[0],baselineData[2])] == 0)) > y.shape[0] / 2:
                rejectNum.append(i)
                # continue
                
            ## reject trials when the NAN includes
            if len(np.argwhere(np.isnan(y[windowL:]) == True)) > 0:
                rejectNum.append(i)
                
            out_y.append(y)
            
        rejectNum = np.unique(rejectNum)
        set(rejectNum)
        y = out_y
    else:
        ## Smoothing
        # y = moving_avg(dat.copy(),windowL)
        y = dat.copy()
        ## baseline(-200ms - 0ms)
        x = np.linspace(TIME_START,TIME_END,y.shape[1])
      
        # filtering
        if len(filt) > 0:
            ave = np.mean(y,axis=1)
            y = y - np.tile(ave, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
            y = butter_bandpass_filter(y, filt[0], filt[1], fs, order=4)
            y = y + np.tile(ave, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
    
        if wid_base.shape[0] > 2:
            tmp_baseline = np.zeros((y.shape[0],y.shape[1]))
            for iTrial in np.arange(wid_base.shape[0]):
                baselineData = np.array([getNearestValue(x,wid_base[iTrial,0]),getNearestValue(x,wid_base[iTrial,1]),getNearestValue(x,wid_analysis)])
                baselinePLR = y[iTrial,np.arange(baselineData[0],baselineData[1])]
                # baselinePLR_std = np.std(baselinePLR,axis=1)
                # baselinePLR_std = np.tile(baselinePLR_std, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
                baselinePLR = np.mean(baselinePLR)
                # baselinePLR = np.tile(baselinePLR, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T   
                tmp_baseline[iTrial,:] = np.tile(baselinePLR, (1,y.shape[1]))
            baselinePLR = tmp_baseline
        else:
            baselineData = np.array([getNearestValue(x,wid_base[0]),getNearestValue(x,wid_base[1]),getNearestValue(x,wid_analysis)])
            baselinePLR = y[:,np.arange(baselineData[0],baselineData[1])]
            baselinePLR_std = np.std(baselinePLR,axis=1)
            baselinePLR_std = np.tile(baselinePLR_std, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
            baselinePLR = np.mean(baselinePLR,axis=1)
            baselinePLR = np.tile(baselinePLR, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T   
       
        if method == 1:
            y = y - baselinePLR
        elif method == 2:
            # y = (y - baselinePLR) / baselinePLR_std
            y = y / baselinePLR
        else:
            ave = np.mean(y,axis=1)
            ave = np.tile(ave, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T
            std = np.std(y,axis=1)
            std = np.tile(std, (1,y.shape[1])).reshape(y.shape[1],y.shape[0]).T           
            y = (y - ave) / std
            
        if cfg['FLAG_LOWPASS']:
            y = lowpass_filter(y, cfg['TIME_END']-cfg['TIME_START'])
       
        rejectNum = reject_trials(y,thres,baselineData)
 

    return y,rejectNum
