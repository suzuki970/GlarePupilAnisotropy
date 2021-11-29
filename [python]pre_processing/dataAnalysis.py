import numpy as np
import matplotlib.pyplot as plt
from pre_processing import pre_processing,re_sampling,moving_avg
from rejectBlink_PCA import rejectBlink_PCA
import json
import os
from pixel_size import pixel_size
import warnings
import matplotlib.patches as patches
import pandas as pd

warnings.simplefilter('ignore')

## ########## initial settings ###################
cfg={
'SAMPLING_RATE':500,   
'windowL':10,
'TIME_START':-1,
'TIME_END':4,
'WID_ANALYSIS':4,
'WID_BASELINE':np.array([-0.2,0]),
'WID_FILTER':np.array([]),
'METHOD':1, #subtraction
'FLAG_LOWPASS':False
}

saveFileLocs = './data/'

cfg['THRES_DIFF'] = 0.04 if cfg['METHOD'] == 1 else 0.1 ## mm

f = open(os.path.join(str('./data/data_original.json')))
dat = json.load(f)
f.close()

mmName = list(dat.keys())

y,rejectNum = pre_processing(np.array(dat['PDR']),cfg)
y = np.delete(y,rejectNum,axis=0)

for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectNum]

## ################## figure plot ##########################
x = np.linspace(cfg['TIME_START'],cfg['TIME_END'],y.shape[1])

plt.figure()
x = np.linspace(cfg['TIME_START'],cfg['TIME_END'],y.shape[1]-1)
plt.subplot(1,2,1)
plt.plot(x,np.diff(y).T)
plt.xlim([cfg['TIME_START'],cfg['TIME_END']])
plt.ylim([-cfg['THRES_DIFF'] ,cfg['THRES_DIFF'] ])
plt.xlabel('Time from response queue')

x = np.linspace(cfg['TIME_START'],cfg['TIME_END'],y.shape[1])

plt.subplot(1,2,2)
plt.plot(x,y.T)
plt.xlim([cfg['TIME_START'],cfg['TIME_END']])
plt.xlabel('Time from response queue')
plt.ylabel('Changes in pupil size')

#### ############### PCA ##########################
pca_x,pca_y,rejectNumPCA = rejectBlink_PCA(y)
y = np.delete(y,rejectNumPCA,axis=0)
for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectNumPCA]

######## rejection of outlier(interplation failed trial) #########
# max_val = [max(abs(y[i,])) for i in np.arange(y.shape[0])]
# fx = np.diff(y)
# rejectOutlier = []
# for i in np.arange(len(y)):
#     if len(np.unique(np.round(fx[i,],5))) < 20:
#         rejectOutlier.append(i)
 
# y = np.delete(y,rejectOutlier,axis=0)
# for mm in mmName:
#     dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectOutlier]
         
# y = re_sampling(y,250)
# gazeX = re_sampling(np.array(dat['gazeX']),250)
# gazeY = re_sampling(np.array(dat['gazeY']),250)
gazeX = np.array(dat['gazeX'])
gazeY = np.array(dat['gazeY'])

################## reject gaze position ##########################
rangeWin = pixel_size(0.369,3,80)
center = np.array([1920, 1080])/2

gazeX = np.mean(gazeX-center[0],axis=1)
gazeY = np.mean(gazeY-center[1],axis=1)

a = rangeWin**2
b = rangeWin**2

tmp_x = gazeX**2
tmp_y = gazeY**2

P = (tmp_x/a)+(tmp_y/b)-1

rejectGaze = np.argwhere(P > 0).reshape(-1)
    
# fig = plt.figure()
# plt.plot(gazeX,gazeY,'.')
# ax = plt.axes()
# e = patches.Ellipse(xy=(0,0), width=rangeWin*2, height=rangeWin*2, fill=False, ec='r')
# ax.add_patch(e)

y = np.delete(y,rejectGaze,axis=0)
for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectGaze]
         
dat['PDR'] = y.tolist()

################## reject subject (N < 40%) ##########################
reject=[]
NUM_TRIAL = 150
numOftrials = []
rejectedTrial = []
for iSub in np.arange(1,int(max(dat['sub']))+1):
    ind = [i for i, sub in enumerate(dat['sub']) if sub == iSub]
    numOftrials.append(len(ind))
    rejectedTrial.append(NUM_TRIAL - len(ind))
    # if numOftrials[iSub-1] < NUM_TRIAL * 0.4:
    #         reject.append(iSub)
            
print('# of trials = ' + str(numOftrials))
print('Averaged # of trials = ' + str(np.round(np.mean(numOftrials),2)))
print('SD # of trials = ' + str(np.round(np.std(numOftrials),2)))

th = NUM_TRIAL - 1.5*np.round(np.std(numOftrials),2)
for iSub in np.arange(1,int(max(dat['sub']))+1):
    if numOftrials[iSub-1] < th:
        reject.append(iSub)

rejectSub = [i for i,d in enumerate(dat['sub']) if d in reject]
print('rejected subject = ' + str(reject))
y = np.delete(y,rejectSub,axis=0)
for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectSub]

rejectedTrial = [d for i,d in enumerate(rejectedTrial) if not i+1 in reject]

ave = np.array(rejectedTrial)/NUM_TRIAL
print('rejected num ave = ' + str(round(np.mean(ave),3)) + ', sd = ' + str(round(np.std(ave),3)))


########################################################################
events = {'sub':[],
          'condition':[],
          'PDR':[],
          'min':[],
          'ave':[]
          }

for iSub in np.unique(dat['sub']):
    for iCond in np.unique(dat['condition']):
        
        ind = np.argwhere((dat['sub'] == iSub ) &
                          (dat['condition'] == np.int64(iCond) )).reshape(-1)
        tmp_y = y[ind,:].mean(axis=0)
        
       
        events['sub'].append(iSub)
        events['condition'].append(iCond)
        events['PDR'].append(tmp_y.tolist())
        
    # events['min'].append(float(x[np.argwhere(tmp_y == np.min(tmp_y[ np.argwhere(x>0)]))]))

plt.figure()

time_min = 0.2
time_max = 4
dat['min'] = []

x_t = x[np.argwhere((x>time_min) & (x<time_max))]
for iSub in np.unique(dat['sub']):
     ind = np.argwhere((events['sub'] == iSub)).reshape(-1)
     
     tmp_p = np.array(events['PDR'])[ind].mean(axis=0)
     tmp_p = tmp_p[np.argwhere(((x>time_min) & (x<time_max)))]
     
     plt.subplot(5,5,iSub)
     plt.plot(x_t,tmp_p)
     
     t = tmp_p.reshape(-1)
     pv = np.gradient(t)
    
     indices = np.where(np.diff(np.sign(pv)))[0]
     xv = np.gradient(indices)
    
     indices = indices[xv > 50]
     
     ev = []
     for itr in np.arange(len(indices)):
         if pv[indices[itr]] - pv[indices[itr]+1] > 0:
             ev.append(1)
         else:
             ev.append(0)
             
     indices = indices[np.argwhere(np.array(ev) == 0).reshape(-1)]
    
     if len(indices)>0:
         indices = indices[0]
         plt.plot(x_t[indices],tmp_p[indices],'ro')  
         dat['min'].append(float(x_t[indices]))
     else:
         dat['min'].append(0)

     # ind = np.argwhere(np.array(dat['sub']) == iSub).reshape(-1)
     # tmp_y = y[ind,indices:]
     # dat['late'] = []
        
########################################################################
# fs = 100
# test_y = moving_avg(y.copy(),fs)
# # test_y = y.copy()
# events = {'sub':[],'condition':[],
#           # 'indices':[],'event':[],
#           'diff_f0':[],'diff_f1':[]}

# plt.figure(figsize=(10,10))
# for iSub in np.unique(dat['sub']):
#     for iCond in np.unique(dat['condition']):
        
#         ind = np.argwhere((dat['sub'] == iSub ) &
#                           (dat['condition'] == np.int64(iCond) )).reshape(-1)
#         tmp_y = test_y[ind,:].mean(axis=0)
        
#         pv = np.gradient(tmp_y)
    
#         indices = np.where(np.diff(np.sign(pv)))[0]
#         xv = np.gradient(indices)
    
#         indices = indices[xv > 50] # < 300ms
        
#         ev = []
#         for itr in np.arange(len(indices)):
#             if pv[indices[itr]] - pv[indices[itr]+1] > 0:
#                 ev.append(1)
#             else:
#                 ev.append(0)
#         ev = np.array(ev)
#         ev = ev[indices > 200]
#         indices = indices[indices > 200]
        
#         if len(indices) > 2:
                  
#             f1 = indices[ev==1]
#             ev = ev[indices >= f1[0]]
#             indices = indices[indices >= f1[0]]
        
#             if len(indices) > 2:
#                 events['sub'].append(int(iSub))
#                 events['condition'].append(int(iCond))
#                 events['diff_f0'].append((tmp_y[indices[1]] - tmp_y[indices[0]]))
#                 events['diff_f1'].append((tmp_y[indices[2]] - tmp_y[indices[1]]))
            
#             # events['indices'].append(indices.tolist())        
#             # events['event'].append(ev.tolist())  
#         # if iSub == 2:
#             plt.subplot(5, 5,iSub)
#             plt.plot(tmp_y)
#             plt.plot(indices[:2],tmp_y[indices[:2]],'ro')    
#             plt.xlim([400,2500])
#             # plt.ylim([-0.2,0.2])

# df = pd.DataFrame.from_dict(events, orient='index').T
# name_loc = ['Upper','Lower','Center','Left','Right',
#             'Upper','Lower','Center','Left','Right']
# name_pat = ['Glare','Glare','Glare','Glare','Glare',
#             'Control','Control','Control','Control','Control']
# df['locs'] = np.array(name_loc)[np.int64(df['condition'].values-1)]
# df['pattern'] = np.array(name_pat)[np.int64(df['condition'].values-1)]

# df = df.groupby(['locs','pattern']).mean()
# a = a.sort_values(['locs'])

# df.plot.bar(y=['diff_f0'], alpha=0.6, figsize=(12,3))
# df.plot.bar(y=['diff_f1'], alpha=0.6, figsize=(12,3))

# with open(os.path.join("./data/events.json"),"w") as f:
#     json.dump(events,f)   
    
## ################## figure plot ##########################
# plt.rcParams["font.size"] = 18
# # 1: top glare, 2: bottom glare, 3: center glare, 4: left glare, 5: right glare; 6: top control, 7: bottom control, 8: center control, 9: left control, 10: right control.
# conditionName = ['glare top' ,'glare bottom','glare center','glare left','glare right',
#                  'control top' ,'control bottom','control center','control left','control left']
# x = np.linspace(cfg['TIME_START'],cfg['TIME_END'],y.shape[1])

# plt.figure(figsize=(20, 6))
# for i in np.arange(2):
#     for j in np.arange(5):
#         cond = i*5+j+1
#         if cond < 6:
#             plt.subplot(1,5,cond)
#         else:
#             plt.subplot(1,5,cond-5)
#         ind = np.argwhere((np.array(dat['condition']) == cond)).reshape(-1)
#         plt.plot(x,np.mean(y[ind,],axis=0),label=conditionName[cond-1])
#         # plt.plot(x,y[ind,].T,alpha=0.2)
                 
# plt.legend(loc='lower right')

del dat['gazeX'], dat['gazeY']
del dat['numOfTrial'], dat['numOfBlink'],dat['numOfSaccade'],dat['ampOfSaccade']

with open(os.path.join("./data/data20211124.json"),"w") as f:
        json.dump(dat,f)
        
        