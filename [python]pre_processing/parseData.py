import numpy as np
import json
from asc2array import asc2array
import glob
import os
from pre_processing import re_sampling
import scipy.io

datHash={"PDR":[],
         "condition":[],
         "gazeX":[],
         "gazeY":[],
         "sub":[],
         "numOfTrial":[],
         "numOfBlink":[],
         "numOfSaccade":[],
         "ampOfSaccade":[],
         "endFix":[]
         }

cfg={'TIME_START':-1,
     'TIME_END':4,
     'THRES_DIFF':10,
     'WID_ANALYSIS':4,
     'useEye':2,
     'WID_FILTER':[],
     'mmFlag':False,
     'normFlag':True,
     's_trg':'SYNCTIME',
     'visualization':False
     }


cfg['TIME'] = cfg['TIME_END']-cfg['TIME_START']

folderName = glob.glob("./results/*")
folderName.sort()
# folderName = folderName[:1]

saveFileLocs = './data/'
numOfSub = 0 

for iSub,subName in enumerate(folderName):
    
    fileName = glob.glob(os.path.join(subName+'/*.mat'))
    mat = scipy.io.loadmat(fileName[0])
    datHash['condition'] = np.r_[datHash['condition'],mat['condFrame'].reshape(-1)]

    fileName = glob.glob(os.path.join(subName+'/*.asc'))
    f = open(os.path.join(str(fileName[0])))
      
    dat=[]
    for line in f.readlines():
        dat.append(line.split())
    f.close()

    eyeData,events,initialTimeVal,fs = asc2array(dat, cfg)
    
    pupilData = eyeData['pupilData']
    gazeX = eyeData['gazeX']
    gazeY = eyeData['gazeY']
    mSaccade = eyeData['mSaccade']
    
    coef = fs / 1000
    
    fix_onset =  [[int((int(e[0]) - initialTimeVal)*coef),e[1]] for e in events['MSG'] if e[1] == 'FIXATION_DiffBase']
    events_onset =  [[int((int(e[0]) - initialTimeVal)*coef),e[1]] for e in events['MSG'] if e[1] == 'Presentation']
    events_offset = [[int((int(e[0]) - initialTimeVal)*coef),e[1]] for e in events['MSG'] if e[1] == 'InterStim']
     
    # print('sub = '+ str(subName[-3:]))
    # print('length = '+ str(len(events_onset)))
    endFix = [e[1:6] for e in events['EFIX'] ]
    
    #%% for heatmap, blink and saccade
    event_data = {'EFIX':[],'ESACC':[],'EBLINK':[]}
    mmName = list(event_data.keys())
    for mm in mmName:
        for i in np.arange(len(events_onset)):       
            tmp = []
            for e in events[mm]:
                if (int(e[1])-initialTimeVal)*coef > fix_onset[i][0] and (int(e[1])-initialTimeVal)*coef < events_offset[i][0]:
                    # if e[0] == 'L':
                    tmp.append(e[1:])
                            
            event_data[mm].append(tmp)
            
    for iTrial,t_onset in zip(event_data['EFIX'],events_onset):
        for e in iTrial:
            e[0] = (int(e[0])-initialTimeVal)*coef-t_onset[0]
            e[1] = (int(e[1])-initialTimeVal)*coef-t_onset[0]
    
    event_data['numOfEBLINK'] = [len(e) for e in event_data['EBLINK']]   
    event_data['numOfESACC'] = [len(e) for e in event_data['ESACC']]
    event_data['ampOfESACC']=[]
    for line in event_data['ESACC']:
        tmp = []
        for e in line:
            tmp.append(float(e[8]))
        event_data['ampOfESACC'].append(np.mean(tmp))
            
    # # ########## data extraction #########    
    for fix,onset,offset in zip(fix_onset,events_onset,events_offset):
        tmp = np.arange(fix[0],offset[0])
        if len(tmp) > cfg['TIME']*fs:
            datHash['PDR'].append(pupilData[tmp])
            datHash['gazeX'].append(gazeX[tmp])
            datHash['gazeY'].append(gazeY[tmp])
        else:
            datHash['PDR'].append(np.zeros(cfg['TIME']*fs))
            datHash['gazeX'].append(np.zeros(cfg['TIME']*fs))
            datHash['gazeY'].append(np.zeros(cfg['TIME']*fs))
       
    datHash['sub'] = np.r_[datHash['sub'], np.ones(len(events_onset))*(numOfSub+iSub+1)]
    
    # datHash['numOfTrial'] = np.r_[datHash['numOfTrial'], tmp]
    datHash['endFix'] = datHash['endFix']+event_data['EFIX']
    datHash['numOfBlink'] = np.r_[datHash['numOfBlink'], np.array(event_data['numOfEBLINK'])]
    datHash['numOfSaccade'] = np.r_[datHash['numOfSaccade'], np.array(event_data['numOfESACC'])]
    datHash['ampOfSaccade'] = np.r_[datHash['ampOfSaccade'], np.array(event_data['ampOfESACC'])]
    
datHash['PDR'] = re_sampling(datHash['PDR'],cfg['TIME']*fs)
datHash['gazeX'] = re_sampling(datHash['gazeX'],cfg['TIME']*fs)
datHash['gazeY'] = re_sampling(datHash['gazeY'],cfg['TIME']*fs)

mmName = list(datHash.keys())
for mm in mmName:
    if not isinstance(datHash[mm],list):
        datHash[mm] = datHash[mm].tolist()
        
with open(os.path.join(saveFileLocs + "data_original.json"),"w") as f:
    json.dump(datHash,f)
       
