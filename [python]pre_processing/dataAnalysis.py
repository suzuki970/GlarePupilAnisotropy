import numpy as np
import matplotlib.pyplot as plt
from pre_processing import pre_processing,re_sampling,moving_avg
from rejectBlink_PCA import rejectBlink_PCA
import json
import os
from pixel_size import pixel_size,pixel2angle
import warnings
import matplotlib.patches as patches
import pandas as pd
from makeEyemetrics import makeMicroSaccade

warnings.simplefilter('ignore')

#%% ###  initial settings ###################
cfg={
'SAMPLING_RATE':500,   
'center':[1920, 1080],  
'DOT_PITCH':0.369,   
'VISUAL_DISTANCE':80,   
'acceptMSRange':2.81,   
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

cfg['THRES_DIFF'] = 0.1 if cfg['METHOD'] == 1 else 0.1 ## mm

f = open(os.path.join(str('./data/data_original.json')))
dat = json.load(f)
f.close()

mmName = list(dat.keys())

y,rejectNum = pre_processing(np.array(dat['PDR']),cfg)
y = np.delete(y,rejectNum,axis=0)

for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectNum]

#%% ###  figure plot ##########################
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

#%% ###  PCA ##########################
# pca_x,pca_y,rejectNumPCA = rejectBlink_PCA(y)
# y = np.delete(y,rejectNumPCA,axis=0)
# for mm in mmName:
#     dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectNumPCA]

#%% ###  rejection of outlier(interplation failed trial) #########
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

#%% ### reject gaze position ###
rangeWin = pixel_size(cfg['DOT_PITCH'],cfg['acceptMSRange'],cfg['VISUAL_DISTANCE'])
center = np.array(cfg['center'])/2

gazeX = moving_avg(np.array(dat['gazeX']).copy(),cfg['SAMPLING_RATE'])
gazeX = re_sampling(gazeX,(cfg['TIME_END']-cfg['TIME_START'])*100)

gazeY = moving_avg(np.array(dat['gazeY']).copy(),cfg['SAMPLING_RATE'])
gazeY = re_sampling(gazeY,(cfg['TIME_END']-cfg['TIME_START'])*100)

gazeX_p = np.mean(gazeX-center[0],axis=1)
gazeY_p = np.mean(gazeY-center[1],axis=1)

gazeX_p=pixel2angle(cfg['DOT_PITCH'],gazeX_p.tolist(),cfg['VISUAL_DISTANCE'])
gazeY_p=pixel2angle(cfg['DOT_PITCH'],gazeY_p.tolist(),cfg['VISUAL_DISTANCE'])

# dat['gazeX'] = gazeX_p.tolist()
# dat['gazeY'] = gazeY_p.tolist()

gazeX = np.mean(gazeX-center[0],axis=1)
gazeY = np.mean(gazeY-center[1],axis=1)

a = rangeWin**2
b = rangeWin**2

tmp_x = gazeX**2
tmp_y = gazeY**2

P = (tmp_x/a)+(tmp_y/b)-1

rejectGaze = np.argwhere(P > 0).reshape(-1)
    
fig = plt.figure()
ax = plt.axes()
e = patches.Ellipse(xy=(0,0), width=rangeWin*2, height=rangeWin*2, fill=False, ec='r')
ax.add_patch(e)
plt.plot(gazeX,gazeY,'.')
plt.plot(gazeX[rejectGaze],gazeY[rejectGaze],'r.')

y = np.delete(y,rejectGaze,axis=0)
for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectGaze]
         
dat['PDR'] = y.tolist()

#%% ### participants reject ###

reject=[]
NUM_TRIAL = 150
numOftrials = []
rejectedTrial = []
for iSub in np.arange(1,int(max(dat['sub']))+1):
    ind = [i for i, sub in enumerate(dat['sub']) if sub == iSub]
    numOftrials.append(len(ind))
    rejectedTrial.append(NUM_TRIAL - len(ind))
    if numOftrials[iSub-1] < NUM_TRIAL * 0.5:
            reject.append(iSub)
            
print('# of trials = ' + str(numOftrials))
print('Averaged # of trials = ' + str(np.round(np.mean(numOftrials),2)))
print('SD # of trials = ' + str(np.round(np.std(numOftrials),2)))


# from scipy.stats import gamma
# import scipy.stats as stats

n, bins, patches = plt.hist(np.array(rejectedTrial))
# a_hat, loc_hat, scale_hat = gamma.fit(n)
# ps_hat = stats.gamma.pdf(bins, a_hat, loc=loc_hat, scale=scale_hat)

# plt.figure()
# plt.plot(ps_hat)

th = np.round(np.std(rejectedTrial),2)
# np.round(np.median(rejectedTrial),2)

# for iSub in np.arange(1,int(max(dat['sub']))+1):
#     if rejectedTrial[iSub-1] > th:
#         reject.append(iSub)


rejectSub = [i for i,d in enumerate(dat['sub']) if d in reject]
print('rejected subject = ' + str(reject))
y = np.delete(y,rejectSub,axis=0)
for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectSub]

rejectedTrial = [d for i,d in enumerate(rejectedTrial) if not i+1 in reject]

ave = np.array(rejectedTrial)/NUM_TRIAL
print('rejected num ave = ' + str(round(np.mean(ave),3)) + ', sd = ' + str(round(np.std(ave),3)))


#%% ##### PC events #############################################
events = {'sub':[],
          'condition':[],
          'PDR':[],
          'min':[]
          }

for iSub in np.unique(dat['sub']):
    for iCond in np.unique(dat['condition']):
        
        ind = np.argwhere((dat['sub'] == iSub ) &
                          (dat['condition'] == np.int64(iCond) )).reshape(-1)
        tmp_y = y[ind,:].mean(axis=0)
        
        tmp_y = moving_avg(tmp_y, 10).reshape(-1)
        events['sub'].append(iSub)
        events['condition'].append(iCond)
        events['PDR'].append(tmp_y.tolist())
  
plt.figure()
time_min = 0.2
time_max = 4
dat['min'] = []
dat['events'] = []
dat['events_p'] = []

x_t = x[np.argwhere((x>time_min) & (x<time_max))]
for iSub in np.unique(dat['sub']):
     ind = np.argwhere((events['sub'] == iSub)).reshape(-1)
     
     tmp_p = np.array(events['PDR'])[ind].mean(axis=0)
     tmp_p = tmp_p[np.argwhere(((x>time_min) & (x<time_max)))]
     
     plt.subplot(5,5,iSub)
     plt.plot(x_t,tmp_p)
     
     t = tmp_p.reshape(-1)
     
     # second-order accurate central differences 
     pv = np.gradient(t)
    
     # find inflection point
     indices = np.where(np.diff(np.sign(pv)))[0]
     # plt.plot(x_t[indices],tmp_p[indices],'bo')  
          
     ev = []
     for itr in np.arange(len(indices)):
         if pv[indices[itr]] - pv[indices[itr]+1] > 0:
             ev.append(1)
         else:
             ev.append(0)
             
     indices = indices[np.argwhere(np.array(ev) == 0).reshape(-1)]
    
     # find inflection point
     xv = np.diff(np.r_[0, indices])
     
     indices = indices[xv > 100]
     
     if np.diff(tmp_p[indices].reshape(-1))[0] < 0:
         indices = indices[1:]
     
     if len(indices)>0:
         dat['events'].append([float(x_t[ind]) for ind in indices.tolist()])
         dat['events_p'].append([float(tmp_p[ind]) for ind in indices.tolist()])
         plt.plot(x_t[indices],tmp_p[indices],'bo')  
         indices = indices[0]
         plt.plot(x_t[indices],tmp_p[indices],'ro')  
         dat['min'].append(float(x_t[indices]))
     else:
         dat['min'].append(0)

     # ind = np.argwhere(np.array(dat['sub']) == iSub).reshape(-1)
     # tmp_y = y[ind,indices:]
     # dat['late'] = []

print('The MPCL was = ' + str(round(np.mean(np.array(dat['min'])),3)) + 
      's, sd = ' + str(round(np.std(np.array(dat['min'])),3)) + 's')


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

# del dat['gazeX'], dat['gazeY']
del dat['numOfTrial'], dat['numOfBlink'],dat['numOfSaccade'],dat['ampOfSaccade']

dat['ampOfMS'] = []
dat['sTimeOfMS'] = []
for iSub in np.unique(dat['sub']):
    ind = np.argwhere(np.array(dat['sub'])==np.int64(iSub)).reshape(-1)
    ev,ms = makeMicroSaccade(cfg,np.array(dat['gazeX'])[ind,:],np.array(dat['gazeY'])[ind,:])
    dat['ampOfMS'] = dat['ampOfMS'] + ms['ampOfMS']
    dat['sTimeOfMS'] = dat['sTimeOfMS'] + ms['sTimeOfMS']
    
plt.figure()
p = np.array(dat['ampOfMS']).mean(axis=0)
plt.plot(p)

# plt.figure(figsize=(10,10))
# tNum=1
# for i,d in enumerate(events[tNum]):

#     # if d[0] == 'x':
#     plt.subplot(3,5,i+1)
#     plt.plot(d[-3],d[-2])
#     plt.title(str(d[1])+'_'+d[0])
    
    # else:
    #     plt.subplot(2,5,i+5)
    #     plt.plot(d[-3],d[-2])
    #     plt.title(str(d[1]))

with open(os.path.join("./data/data20211124.json"),"w") as f:
        json.dump(dat,f)
