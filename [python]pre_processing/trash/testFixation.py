#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 15:32:37 2021

@author: yutasuzuki
"""

gazeX = np.array(dat['gazeX'])
gazeY = np.array(dat['gazeY'])

gazeX = zeroInterp(gazeX.copy(),500,5)
gazeX = gazeX['pupilData']

gazeY = zeroInterp(gazeY.copy(),500,5)
gazeY = gazeY['pupilData']

gv = [list(np.gradient(g[500:])*500) for g in gazeX.tolist()]
gacc = [list(np.gradient(np.gradient(g[500:]))*(500**2)) for g in gazeX.tolist()]

thFix = pixel_size(cfg['DOT_PITCH'],30,cfg['VISUAL_DISTANCE'])
thAccFix = thFix*400

itl = 20

fix=dat['endFix'][itl]

sigma = np.std(gv[itl])
sigma = sigma*3

sigma_acc = np.std(gacc[itl])
sigma_acc = sigma_acc*3

ind = np.argwhere(abs(np.array(gv[itl])) > sigma).reshape(-1)
ind_acc = np.argwhere(abs(np.array(gacc[itl])) > sigma_acc).reshape(-1)
# ind_acc = ind_acc[np.argwhere(np.diff(np.r_[0, ind_acc]) > 10)].reshape(-1)
# ind = np.unique(np.r_[ind,ind_acc])

if np.max(np.diff(np.r_[0, ind])) > 1:
    eFixTime = ind[np.argwhere(np.diff(np.r_[0, ind]) > 10)].reshape(-1)
    if len(eFixTime) == 0:
        eFixTime = ind[0]
    eFixTime = np.r_[eFixTime,len(gv[itl])]

    sFixTime = ind[np.r_[np.argwhere(np.diff(np.r_[0, ind]) > 10)[1:].reshape(-1)-1,len(ind)-1]]
    sFixTime = np.r_[0,sFixTime]
    
    tmp_endFix = []
    for iFix in np.arange(len(sFixTime)):
        tmp_endFix.append([0,0,0,
                           gazeX[itl,np.arange(sFixTime[iFix],eFixTime[iFix])].mean(),
                           gazeY[itl,np.arange(sFixTime[iFix],eFixTime[iFix])].mean(),
                           0])

else:
    sFixTime = ind[0].tolist()
    eFixTime = ind[-1].tolist()
    tmp_endFix = [0,0,0,
                  gazeX[itl,np.arange(sFixTime,eFixTime)].mean(),
                  gazeY[itl,np.arange(sFixTime,eFixTime)].mean(),
                  0]


#%% coordinate x
plt.figure(figsize=(8,10))
plt.subplot(3,2,1)
plt.plot(gazeX[itl,500:],linewidth = 0.3)

for ifix in fix:
    plt.plot(fix[itl,500:],linewidth = 0.3)


for iFix in np.arange(len(sFixTime)):
    plt.plot(np.arange(sFixTime[iFix],eFixTime[iFix]),
             gazeX[itl,np.arange(sFixTime[iFix],eFixTime[iFix])+500],
             linewidth = 0.3,linestyle='solid')

plt.subplot(3,2,2)
plt.hist(np.gradient(gazeX[itl,500:]),bins=100)

#%% velocity of x
plt.subplot(3,2,3)
plt.plot(gv[itl])
plt.plot(ind,np.array(gv[itl])[ind],'.r')

plt.subplot(3,2,4)
plt.hist(np.array(gv)[itl,:],bins=100)

plt.axvline(x=thFix, ls = "--", color='#2ca02c', alpha=0.7)
plt.axvline(x=-thFix, ls = "--", color='#2ca02c', alpha=0.7)
plt.axvline(x=sigma, ls = "--", color='red', alpha=0.7)
plt.axvline(x=-sigma, ls = "--", color='red', alpha=0.7)

#%% acceration of x
plt.subplot(3,2,5)
plt.plot(gacc[itl])

plt.plot(ind,gazeX[itl,ind],'.r')

plt.subplot(3,2,6)
plt.hist(np.array(gacc)[itl,:],bins=100)

plt.axvline(x=thAccFix, ls = "--", color='#2ca02c', alpha=0.7)
plt.axvline(x=-thAccFix, ls = "--", color='#2ca02c', alpha=0.7)
plt.axvline(x=sigma_acc, ls = "--", color='red', alpha=0.7)
plt.axvline(x=-sigma_acc, ls = "--", color='red', alpha=0.7)
