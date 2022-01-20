import wave
import numpy as np
import struct
import sys
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile
import librosa.display 
from pydub import AudioSegment
import glob
import os
import random   
from scipy.signal import chirp
import soundfile as sf

class makeAuditoryStim:    
    def __init__(self, cfg):
        self.freq_pattern = cfg['freq_pattern']
        self.spl = cfg['vol_pattern']
        self.sec = cfg['sec']
        self.isi = cfg['isi']
        self.fs = cfg['fs']
        self.keepFile = cfg['keepFile']
        self.sepFlg = cfg['sep']
        self.revFlg = cfg['reverse']
        self.iter = cfg['iter']
        self.falltime = cfg['fallTime']
        self.cfg = cfg
        
        if not os.path.exists(cfg['saveFolder']):
            os.mkdir(cfg['saveFolder'])
        
    #%%
    def makeSweepSound(self,freqWin,vol):
       
        filepath = self.cfg['saveFolder'] + '/sweep'
        
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            fileListRemove = glob.glob(filepath + '/*.wav')
            for f in fileListRemove:
                os.remove(f)
                
                
        t = np.linspace(0, self.sec, int(self.fs * self.sec))
        
        b1 = chirp(t, f0=freqWin[0], f1=freqWin[1], t1=self.sec, method='linear')
        
        spl = 10**(vol/20)*0.4
        max_num = (32767.0*spl) / max(abs(b1)) # -32768~32767(int16)
        
        wave16 = [int(x * max_num) for x in b1] # transform 16bit
                   
        bi_wave = struct.pack("h" * len(wave16), *wave16) # binary format
        fileName = filepath + '/SweepSound_' + str(freqWin[0]) + '_' + str(freqWin[1]) + '_SPL' + str(vol) + '.wav'
        
        w = wave.Wave_write(fileName)  # output
        
        p = (1, 2, self.fs, len(bi_wave), 'NONE', 'not compressed')
        
        w.setparams(p) 
        w.writeframes(bi_wave)
        w.close()
            
        sound = AudioSegment.from_file(fileName, "wav")
        sound1 = sound.fade_in(self.falltime).fade_out(self.falltime)
        sound1.export(filepath + '/SweepSound_fadeinout_' + str(freqWin[0]) + '_' + str(freqWin[1]) + '_SPL' + str(vol) + '.wav', format="wav")
        
        # plt.figure()         
        # plt.plot(t, b1)
    #%%
    def repeatTones(self,fPath,repTimes):
        # filepath = self.cfg['saveFolder'] + '/repeat'
        
        fileList = glob.glob(fPath + '/*.wav')
        fileList.sort()
        
        filepath = self.cfg['saveFolder'] + '/repeat'
        
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            fileListRemove = glob.glob(filepath + '/*.wav')
            for f in fileListRemove:
                os.remove(f)
     
        for i,f in enumerate(fileList):
            s = wave.open(f, 'r')
            data = s.readframes(s.getnframes())
            s.close()
         
            bl = np.frombuffer(data, dtype=np.int16)

            bl = np.tile(bl,repTimes)
            w = np.array([bl,bl]).transpose()
            data = w.astype(np.int16)
       
            out = wave.open(filepath + '/repear_' + f.split('/')[-1],'w')
        
            out.setnchannels(2) #stereo
            out.setsampwidth(2) #16bits
            out.setframerate(self.fs)
            out.writeframes(data.tostring())
            out.close()
      
     
    #%%
    def composeTones(self,repTimes):
        filepath = self.cfg['saveFolder'] + '/fadeinout'
        
        fileList = glob.glob(filepath + '/*.wav')
        fileList.sort()
        
        filepath = self.cfg['saveFolder'] + '/composed'
        
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            fileListRemove = glob.glob(filepath + '/*.wav')
            for f in fileListRemove:
                os.remove(f)
    
        for i,f in enumerate(fileList):
        
            s = wave.open(f, 'r')
            data = s.readframes(s.getnframes())
            s.close()
        
            wave16 = np.frombuffer(data, dtype=np.int16)
            
            wave16 = np.tile(wave16,repTimes)
            bi_wave = struct.pack("h" * len(wave16), *wave16) # binary format
          
            p = (1, 2, self.fs, len(bi_wave), 'NONE', 'not compressed')
      
            w = wave.Wave_write(filepath + '/' + f.split('/')[3])
           
            w.setparams(p) 
            w.writeframes(bi_wave)
            w.close()
    #%%
    def synthesizeTones(self):
        
        filepath = self.cfg['saveFolder'] + '/synthesize'
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            fileListRemove = glob.glob(filepath + '/*.wav')
            for f in fileListRemove:
                os.remove(f)
                
        t = np.arange(0, self.fs * self.sec) 
        b1=[]
        for i,(f,s) in enumerate(zip(self.freq_pattern,self.spl)):
            spl = 10**(s/20)*0.4
            b1.append(np.sin(2 * np.pi * f * t / self.fs))

        b1=np.array(b1).sum(axis=0)
        
        max_num = (32767.0) / max(abs(b1)) # -32768~32767(int16)
        
        wave16 = [int(x * max_num) for x in b1] # transform 16bit
        plt.plot(wave16)
                   
        bi_wave = struct.pack("h" * len(wave16), *wave16) # binary format
                
        w = wave.Wave_write(filepath + '/synthesized.wav')  # output
        
        p = (1, 2, self.fs, len(bi_wave), 'NONE', 'not compressed')
        
        w.setparams(p) 
        w.writeframes(bi_wave)
        w.close()
        
     #%%    
    def makePureTones(self):
        t = np.arange(0, self.fs * self.sec) 
        
        filepath = self.cfg['saveFolder'] + '/original'
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            fileListRemove = glob.glob(filepath + '/*.wav')
            for f in fileListRemove:
                os.remove(f)
       
        for i,(f,s) in enumerate(zip(self.freq_pattern,self.spl)):
            spl = 10**(s/20)*0.4
            # max_num = (32767.0*spl) / max(abs(sine_wave)) # -32768~32767(int16)
    
            b1 = np.sin(2 * np.pi * f * t / self.fs)
            
            # plt.figure()
            # t = np.arange(0, 200000 * 0.075) 
            # plt.plot(np.sin(2 * np.pi * 20000 * t / 200000))
            # plt.xlim([0,50])
            # spl=1
            max_num = (32767.0*spl) / max(abs(b1)) # -32768~32767(int16)
            
            wave16 = [int(x * max_num) for x in b1] # transform 16bit
                       
            bi_wave = struct.pack("h" * len(wave16), *wave16) # binary format
                    
            if i < 10:
                w = wave.Wave_write(filepath + '/0' + str(i) + '_' + str(f) + '.wav')  # output
            else:
                w = wave.Wave_write(filepath + '/' + str(i)  + '_' + str(f) + '.wav')  # output
            
            p = (1, 2, self.fs, len(bi_wave), 'NONE', 'not compressed')
            
            w.setparams(p) 
            w.writeframes(bi_wave)
            w.close()
            
            if i < 10:
                w0, fs = sf.read(filepath + '/0' + str(i) + '_' + str(f) + '.wav')
                rms = librosa.feature.rms(y=w0)
                times = librosa.times_like(rms, sr=fs)
                plt.plot(times, 20 * np.log10(rms[0]*2**(1/2)))
        
        
    def adjustFadein_out(self,fPath):
     
        fileList = glob.glob(fPath + '/*.wav')
        fileList.sort()
        
        filepath = fPath.split('/')[:-2]
        filepath = '/'.join(filepath) + '/fadeinout'
        
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            fileListRemove = glob.glob(filepath + '/*.wav')
            for f in fileListRemove:
                os.remove(f)
                
        for i,f in enumerate(fileList):
        
            sound = AudioSegment.from_file(f, "wav")
            sound1 = sound.fade_in(self.falltime).fade_out(self.falltime)
            
            if i < 10:
                sound1.export(filepath + '/0' + str(i) + '_' + str(self.freq_pattern[i]) + 
                              '_SPL' + str(self.spl[i]) + '.wav', format="wav")
            else:
                sound1.export(filepath + '/' + str(i) + '_' + str(self.freq_pattern[i]) + 
                              '_SPL' + str(self.spl[i]) + '.wav', format="wav")
        
        # if not self.keepFile:
        #     for f in fileList:
        #         os.remove(f)
            
    def makeStreamSound(self):
        
        fileList = glob.glob('./*fadeinout*.wav')
        fileList.sort()
        b = []
        for i,f in enumerate(fileList):
            s = wave.open(f, 'r')
            data = s.readframes(s.getnframes())
            s.close()
        
            b.append(np.frombuffer(data, dtype=np.int16))

        if self.sepFlg:
            br = np.concatenate([b[0],
                                 np.zeros(int(self.isi*self.fs)),
                                 np.zeros(int(self.sec*self.fs)),
                                 np.zeros(int(self.isi*self.fs)),
                                 b[0],
                                 np.zeros(int(self.isi*self.fs)),
                                 np.zeros(int(self.sec*self.fs)),
                                 np.zeros(int(self.isi*self.fs))])
            
            bl = np.concatenate([np.zeros(int(self.sec*self.fs)),
                                 np.zeros(int(self.isi*self.fs)),
                                 b[1],
                                 np.zeros(int(self.isi*self.fs)),
                                 np.zeros(int(self.sec*self.fs)),
                                 np.zeros(int(self.isi*self.fs)),
                                 np.zeros(int(self.sec*self.fs)),
                                 np.zeros(int(self.isi*self.fs))])
        else:
            br = np.concatenate([b[0],
                                  np.zeros(int(self.isi*self.fs)),
                                  b[1],
                                  np.zeros(int(self.isi*self.fs)),
                                  b[0],
                                  np.zeros(int(self.isi*self.fs)),
                                  np.zeros(int(self.sec*self.fs)),
                                  np.zeros(int(self.isi*self.fs))])
            
            bl = np.concatenate([b[0],
                                  np.zeros(int(self.isi*self.fs)),
                                  b[1],
                                  np.zeros(int(self.isi*self.fs)),
                                  b[0],
                                  np.zeros(int(self.isi*self.fs)),
                                  np.zeros(int(self.sec*self.fs)),
                                  np.zeros(int(self.isi*self.fs))])
       
        if self.revFlg:
           t = bl.copy()
           bl = br.copy()
           br = t.copy()
            
        bl = np.tile(bl,self.iter)
        br = np.tile(br,self.iter)
    
        w = np.array([br,bl]).transpose()
        data = w.astype(np.int16)
        
        if self.sepFlg:
            out = wave.open('./stereo_sep.wav','w')
        else:
            out = wave.open('./stereo.wav','w')
        
        out.setnchannels(2) #stereo
        out.setsampwidth(2) #16bits
        out.setframerate(self.fs)
        out.writeframes(data.tostring())
        out.close()
      
        if not self.keepFile:
            for f in fileList:
                os.remove(f)
        
   
    def makeRandSound(self):
        
        fileList = glob.glob('./*fadeinout*.wav')
        fileList.sort()
        b = []
        for i,f in enumerate(fileList):
            s = wave.open(f, 'r')
            data = s.readframes(s.getnframes())
            s.close()     
            b.append(np.frombuffer(data, dtype=np.int16))

        arraylist = list(np.arange(6))
        random.shuffle(arraylist)
        arraylist = []
        for i in np.arange(5):
            arraylist.append([random.randint(0,5),
                              random.randint(0,5),
                              random.randint(0,5)])        
        br = []
        bl = []
        # for ind in arraylist:
        #     # br = np.r_[br,b[ind]]
        #     # br = np.r_[br,np.zeros(int(self.isi*self.fs))]
        #     br = np.concatenate([br,
        #                         b[ind[0]],
        #                         np.zeros(int(self.isi*self.fs)),
        #                         # b[ind[1]],
        #                         np.zeros(int(self.sec*self.fs)),
        #                         np.zeros(int(self.isi*self.fs)),
        #                         b[ind[2]],
        #                         np.zeros(int(self.isi*self.fs)),
        #                         np.zeros(int(self.sec*self.fs)),
        #                         np.zeros(int(self.isi*self.fs))])
        # for _ in np.arange(5):
        #     br = np.concatenate([br,
        #                           b[1],
        #                           np.zeros(int(self.isi*self.fs)),
        #                           b[0],
        #                           np.zeros(int(self.isi*self.fs)),
        #                           b[1],
        #                           np.zeros(int(self.isi*self.fs)),
        #                           np.zeros(int(self.sec*self.fs)),
        #                           np.zeros(int(self.isi*self.fs))])
        for _ in np.arange(10):
            br = np.concatenate([br,
                                 b[0],
                                 np.zeros(int(self.isi*self.fs)),
                                 b[1],
                                 np.zeros(int(self.isi*self.fs)),
                                 b[0],
                                 np.zeros(int(self.isi*self.fs)),
                                 np.zeros(int(self.sec*self.fs)),
                                 np.zeros(int(self.isi*self.fs))])
            
        for ind in arraylist:
            br = np.concatenate([br,
                                  b[0],
                                  np.zeros(int(self.isi*self.fs)),
                                  np.zeros(int(self.sec*self.fs)),
                                  np.zeros(int(self.isi*self.fs)),
                                  b[0],
                                  np.zeros(int(self.isi*self.fs)),
                                  np.zeros(int(self.sec*self.fs)),
                                  np.zeros(int(self.isi*self.fs))])
        
        for _ in np.arange(5):
            br = np.concatenate([br,
                                 b[0],
                                 np.zeros(int(self.isi*self.fs)),
                                 b[1],
                                 np.zeros(int(self.isi*self.fs)),
                                 b[0],
                                 np.zeros(int(self.isi*self.fs)),
                                 np.zeros(int(self.sec*self.fs)),
                                 np.zeros(int(self.isi*self.fs))])
                   
           # bl = np.r_[bl,b[ind]]
           # bl = np.r_[bl,np.zeros(int(self.isi*self.fs))]
      
       
        bl=br.copy()   
        # bl = np.tile(bl,self.iter)
        # br = np.tile(br,self.iter)
    
        w = np.array([br,bl]).transpose()
        data = w.astype(np.int16)
        
        if self.cfg['sep']:
            out = wave.open('./stereo_sep.wav','w')
        else:
            out = wave.open('./stereo.wav','w')
        
        out.setnchannels(2) #stereo
        out.setsampwidth(2) #16bits
        out.setframerate(self.fs)
        out.writeframes(data.tostring())
        out.close()
      
        if not self.cfg['fileKeep']:
            for f in fileList:
                os.remove(f)
    

