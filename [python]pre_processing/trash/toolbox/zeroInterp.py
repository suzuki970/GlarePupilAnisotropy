
# import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from pre_processing import moving_avg
                    
def zeroInterp(pupilData,fs,interval,windowL=0):
   
    if pupilData.ndim == 1:
        pupilData = pupilData.reshape(1,len(pupilData))       
        
    interpolatedArray=[]
    for iTrial in np.arange(pupilData.shape[0]): 

        y_base = pupilData[iTrial,].copy()
        y_base = y_base[windowL:len(y_base)]
        
        y = pupilData[iTrial,].copy()
        if windowL > 0:
            y = moving_avg(y,windowL).reshape(-1)
        
        y[np.argwhere(y < 10**-5)] = 0
        
        zeroInd = np.argwhere(y[0:-1]==0).reshape(-1)
        
        if len(zeroInd) == len(y)-1:
            continue
        
        if len(zeroInd) > 0:
            
            diffOnOff = np.diff(zeroInd)
            diffOnOff = np.append(diffOnOff,10**10)
            diffOnOff = np.append(10**10,diffOnOff)
            count=0
            datOfblinkCood=[]
           
            for i in np.arange(1,len(diffOnOff)):
    
                if diffOnOff[i] >= interval and diffOnOff[i - 1] >= interval:  #### one-shot noise
                    datOfblinkCood.append(np.array([zeroInd[i - 1],zeroInd[i - 1]]))
                    count=count + 1
        
                elif diffOnOff[i] >= interval and diffOnOff[i - 1] <= interval:
                    datOfblinkCood[count][1] = zeroInd[i - 1]
                    count=count + 1
        
                elif diffOnOff[i] < interval and diffOnOff[i - 1] < interval:
                    pass
                elif diffOnOff[i] < interval and diffOnOff[i - 1] >= interval:
                    datOfblinkCood.append(np.array([zeroInd[i - 1],0]))
             
            ###### adjust the onset and offset of the eye blinks
            for i in np.arange(len(datOfblinkCood)):
                # for onset
                while (y[datOfblinkCood[i][0]] - y[datOfblinkCood[i][0]-1]) <= 0:
        
                    datOfblinkCood[i][0] = datOfblinkCood[i][0]-1
                   
                    if datOfblinkCood[i][0] == 0:
                        break
        
                # for offset
                while (y[datOfblinkCood[i][1]] - y[datOfblinkCood[i][1]+1]) <= 0:
        
                    datOfblinkCood[i][1] = datOfblinkCood[i][1]+1
        
                    if datOfblinkCood[i][1] == len(y)-1:
                        break
                    
            for db in datOfblinkCood:
                onsetArray = db[0]
                offsetArray = db[1]
                if onsetArray == offsetArray:
                   y_base[onsetArray]=0
                else:
                   y_base[onsetArray:offsetArray]=0
              
            y = np.r_[pupilData[iTrial,np.arange(windowL)],y_base]
            nonZero = np.argwhere(y != 0).reshape(-1)
            if len(nonZero) != len(y):
                numX = np.arange(len(y))
                yy = interpolate.PchipInterpolator(numX[nonZero], pupilData[iTrial,nonZero])
       
                pupilData[iTrial,] = yy(np.arange(len(y)))
            
            interpolatedArray.append(len(y)-len(nonZero))
        else:
            interpolatedArray.append(0)
            
    data_frame = {'pupilData':pupilData,
                  'interpolatedArray':interpolatedArray}
    
    return data_frame

   