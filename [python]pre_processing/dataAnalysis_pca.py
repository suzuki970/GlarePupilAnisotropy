#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:07:26 2021

@author: yutasuzuki
"""
import numpy as np
import matplotlib.pyplot as plt
from pre_processing import pre_processing,re_sampling
from rejectBlink_PCA import rejectBlink_PCA
import json
import os
from pixel_size import pixel_size
import warnings
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
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

saveFileLocs = '../data/'

cfg['THRES_DIFF'] = 0.04 if cfg['METHOD'] == 1 else 0.1 ## mm

f = open(os.path.join(str('./data_original.json')))
dat = json.load(f)
f.close()

mmName = list(dat.keys())

# ################## pre-pro ##########################
y,rejectNum = pre_processing(np.array(dat['PDR']),cfg)
y = np.delete(y,rejectNum,axis=0)

for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectNum]

# ################## PCA ##########################
# pca_x,pca_y,rejectNumPCA = rejectBlink_PCA(y)
# y = np.delete(y,rejectNumPCA,axis=0)
# for mm in mmName:
#     dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectNumPCA]

# ################## rejection of outlier(interplation failed trial) ##########################
max_val = [max(abs(y[i,])) for i in np.arange(y.shape[0])]
fx = np.diff(y)
rejectOutlier = []
for i in np.arange(len(y)):
    if len(np.unique(np.round(fx[i,],5))) < 20:
        rejectOutlier.append(i)
 
y = np.delete(y,rejectOutlier,axis=0)
for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectOutlier]

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
    
fig = plt.figure()
plt.plot(gazeX,gazeY,'.')
ax = plt.axes()
e = patches.Ellipse(xy=(0,0), width=rangeWin*2, height=rangeWin*2, fill=False, ec='r')
ax.add_patch(e)

y = np.delete(y,rejectGaze,axis=0)
for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectGaze]
         
dat['PDR'] = y.tolist()

################## reject subject (N < 40%) ##########################
reject=[]
NUM_TRIAL = 150
numOftrials = []
for iSub in np.arange(1,int(max(dat['sub']))+1):
    ind = [i for i, sub in enumerate(dat['sub']) if sub == iSub]
    numOftrials.append(len(ind))
    
    if numOftrials[iSub-1] < NUM_TRIAL * 0.6:
            reject.append(iSub)
print('# of trials = ' + str(numOftrials))

rejectSub = [i for i,d in enumerate(dat['sub']) if d in reject]
print('rejected subject = ' + str(reject))
y = np.delete(y,rejectSub,axis=0)  
for mm in mmName:
    dat[mm] = [d for i,d in enumerate(dat[mm]) if not i in rejectSub]
        
################## PCA score ##########################

y = y[:,400:]
numOfcomp = 2
pca = PCA(n_components=numOfcomp).fit(y)
values = pca.transform(y)

loadings = pca.components_  # Eigenvector    
var_ratio = pca.explained_variance_ratio_
    
x = np.linspace(-0.2,4,y.shape[1])
plt.figure(figsize=(10, 7))
for i in np.arange(numOfcomp):
    plt.plot(x,loadings[i,],'-',label=round(var_ratio[i],3))
        # plt.plot(x,tmp.mean(axis=0),'-',label=round(var_ratio[i],3))
plt.legend()  

dat_loadings = {'sub':[],'locs':[],'compornent':[],'pattern':[],'cnum':[]}
for iCond in np.arange(1,6):
    data_pupil = {}
    for iSub in np.unique(dat['sub']):  
        #### Glare #####
        ind_glare = np.argwhere((np.array(dat['sub']) == np.int64(iSub)) &
                      (np.array(dat['condition']) == iCond)).reshape(-1)
        ##### Control #####
        ind_control = np.argwhere((np.array(dat['sub']) == np.int64(iSub)) &
                      (np.array(dat['condition']) == iCond+5)).reshape(-1)
        
        for icnum in np.arange(numOfcomp):
              dat_loadings['cnum'].append(int(icnum)) 
              dat_loadings['compornent'].append(values[ind_glare,icnum].mean())
              dat_loadings['sub'].append(int(iSub))            
              dat_loadings['locs'].append(int(iCond))
              dat_loadings['pattern'].append(0)
         
              dat_loadings['cnum'].append(int(icnum)) 
              dat_loadings['compornent'].append(values[ind_control,icnum].mean())
              dat_loadings['sub'].append(int(iSub))            
              dat_loadings['locs'].append(int(iCond))
              dat_loadings['pattern'].append(1)
        
with open(os.path.join("../data/dat_loadings.json"),"w") as f:
    json.dump(dat_loadings,f)
    
# df = pd.DataFrame.from_dict(dat_loadings, orient='index').T

# name_loc = ['Upper','Lower','Center','Left','Right']
# name_pat = ['Glare','Control']
# df['locs'] = np.array(name_loc)[np.int64(df['locs'].values-1)]
# df['pattern'] = np.array(name_pat)[np.int64(df['pattern'].values)]

# for icnum in np.arange(numOfcomp):
#     dat_plot = df[df['cnum'] == icnum]
#     dat_plot = dat_plot.drop('cnum', axis=1)
    
#     dat_plot = dat_plot.groupby(['locs','pattern']).mean()
#     # a = a.sort_values(['locs'])
    
#     dat_plot.plot.bar(y=['compornent'], alpha=0.6, figsize=(12,3))

      
# plt.legend(loc='lower right',fontsize=14,title=gName[iCond-1])
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=12,title="proportion of the variance")
# lg = plt.legend(loc='upper left',title='proportion of the variance')
# lg.get_title().set_fontsize(16)
# bbox_to_anchor=(1.05, 1), 
# plt.legend()
# plt.ylabel('Loadings')
# plt.xlabel('Time [sec]')


# plt.figure(figsize=(8, 6))
# plt.rcParams["font.size"] = 18
# numOfsub = len(np.unique(dat['sub']))
# plt.errorbar(gName, summary_pca.groupby(["condition"]).mean()['values'], 
#               yerr = summary_pca.groupby(["condition"]).std()['values']/np.sqrt(numOfsub), 
#               capsize=5, fmt='o', markersize=10, ecolor='black', 
#               markeredgecolor = "black", 
#               color='w')