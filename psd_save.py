import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
from collections import Counter
import mne
import scipy
from scipy.integrate import simpson
import concurrent.futures


def return_bandpower(eeg_epoch,sampfreq,winlen):
  feats = []
  
  win = winlen * sampfreq
  freqs, psd = scipy.signal.welch(eeg_epoch, sampfreq, nperseg=win)
  
  psd_range = psd[np.logical_and(freqs > 0, freqs <= 20)]

  #print(psd[np.logical_and(freqs > 0, freqs <= 20)].shape)
  
  
  return psd_range
    
def save_psd(channel,whichstage):

  m1 = '/media/ayush/2ed9ed77-f1c6-421b-82ba-49d09743f1f0/BCH/'
  
  
  finame='/home/ayush/Documents/BCH_data_analysis/Demographics/Demographics.csv'
  
  df = pd.read_csv(finame)
  ID = df['BIDS'].to_numpy()
  sess = df['Session'].to_numpy()
  agedays = df['AgeDays'].to_numpy()
  
  name_counts = {}
  
  
  ctr = 0
  wake = np.zeros((15695,40))
  ageyrs = np.zeros((15695,))
  
  
  for i in range(0,len(sess)):
  
      for fi in os.listdir(os.path.join(m1,ID[i],'ses-'+str(sess[i]),'eeg')):
      
        if fi.endswith('_sleepannotations.csv'):
          df = pd.read_csv(os.path.join(m1,ID[i],'ses-'+str(sess[i]),'eeg',fi))
          df['sleep_stage'] = df['sleep_stage'].replace('N4','N3')
          #print(os.path.join(m1,ID[i],'ses-'+str(sess[i]),'eeg',fi))
          
        if fi.endswith('.edf'):
          if channel=='C3' or channel=='F3' or channel=='O1'
            raw = mne.io.read_raw_edf(os.path.join(m1,ID[i],'ses-'+str(sess[i]),'eeg',fi), include = [channel,'M2','A2'], verbose='ERROR')
          if channel=='C4' or channel=='F4' or channel=='O2'
            raw = mne.io.read_raw_edf(os.path.join(m1,ID[i],'ses-'+str(sess[i]),'eeg',fi), include = [channel,'M1','A1'], verbose='ERROR')
          
          raw.load_data(verbose=False)
          ext_signal = raw.get_data().T
          specific_signal = ext_signal[:,0] - ext_signal[:,1]
          #print(specific_signal.shape)
          params = {'Fs': raw.info['sfreq']} 
          sampf = params['Fs']
                
        wake_df = df[df['sleep_stage'] == whichstage]
        start_end_times = wake_df[['sample_stamp_start', 'sample_stamp_end']].values
        wake_psd = np.zeros((len(start_end_times),40))
        psd_ctr = 0
          
        for start, end in start_end_times:
          start_idx = int(start)-1
          end_idx = int(end)
          
          if end_idx<=len(specific_signal):
              segment = specific_signal[start_idx:end_idx]
              resampled_segment = scipy.signal.resample(segment, int(len(segment)*200/sampf) )
              #print(len(resampled_segment))
              wake_psd[psd_ctr,:] = return_bandpower(resampled_segment,sampfreq=200,winlen=2)
              psd_ctr = psd_ctr+1
              
        if psd_ctr!=0:
          #print(psd_ctr)
          #print(wake_psd.shape)
          #print(np.mean(wake_psd[:,0:psd_ctr],axis=0).shape)
          wake[ctr,:] = np.mean(wake_psd[0:psd_ctr,:],axis=0)
          ageyrs[ctr] = agedays[i]/365
          ctr = ctr + 1
          print(whichstage + ' ' + channel + ' ' + str(ctr))
          

          
 
  np.save(whichstage + '_' + channel + '.npy',wake[0:ctr,:])
  np.save(whichstage + '_' + channel + '_age.npy',ageyrs[0:ctr])
  
  
channels = ['C3', 'F3', 'O1','C4', 'F4', 'O2']
stages = ['N1', 'N2', 'N3', 'REM', 'WAKE']

def run_in_parallel(args):
  channel, whichstage = args
  save_psd(channel, whichstage)

# Create a list of arguments for each combination of channel and whichstage
arguments = [(channel, whichstage) for channel in channels for whichstage in stages]

# Use ThreadPoolExecutor to run in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
  # Map the function to the arguments
  executor.map(run_in_parallel, arguments)
