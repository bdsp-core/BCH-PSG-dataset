import numpy as np
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
from collections import Counter
import seaborn as sns

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.special import softmax

##########################################################################################################
# NEED TO RUN THE SCRIPT psd_save.py first to extact the PSD
##########################################################################################################

def return_avgpsd(region,whichstage,minage,maxage):

    if region=='frontal':
        ch1 = 'F3'
        ch2 = 'F4'
    elif region=='central':
        ch1 = 'C3'
        ch2 = 'C4'
    elif region=='occipital':
        ch1 = 'O1'
        ch2 = 'O2'
    
    left = np.load(whichstage+'_' + ch1 + '.npy')
    left_age = np.load(whichstage+'_' + ch1 + '_age.npy')
    right = np.load(whichstage+'_' + ch2 + '.npy')
    right_age = np.load(whichstage+'_' + ch2 + '_age.npy')
    avg_age = (left_age+right_age)/2
    avg_psd = (left+right)/2
    
    psd_interest = np.zeros((len(left_age),40))
    final_age  = np.zeros((len(left_age),))
    
    ctr = 0
    
    for i in range(0,len(left_age)):
        if np.min(avg_psd[i,:])!=0 and avg_age[i]>=minage and avg_age[i]<maxage:
          psd_interest[ctr,:] = 10*np.log10(avg_psd[i,:]*1000000000000)
          final_age[ctr] = avg_age[i]
          ctr = ctr + 1
          
    psd_interest = psd_interest[0:ctr,:]
    final_age = final_age[0:ctr]
    
    psd_final = np.mean(psd_interest,axis=0)
    #print(psd_final.shape)
    return psd_final


minage_arr = [0,0.5,1,2,4,6,8,10,12,14,16]
maxage_arr = [0.5,1,2,4,6,8,10,12,14,16,18]
frequencies = np.linspace(0, 20, 40)
#palette = sns.color_palette('magma',len(minage_arr))
cmap = plt.get_cmap('magma')
palette = cmap(np.linspace(0.2,0.9,len(minage_arr)))

for stages in ['WAKE','N1','N2','N3','REM']: 
    for regions in ['occipital','frontal','central']:
    
        print(stages)
    
        show_xlabel = False
        show_ylabel = False
        
        if regions=='frontal':
            show_ylabel=True
        if stages=='REM':
            show_xlabel=True

        plt.figure(figsize=(10, 7))
        
        for i in range(0,len(minage_arr)):
            
            plt.plot(frequencies, return_avgpsd(regions,stages,minage_arr[i],maxage_arr[i]), label=str(minage_arr[i]) + '-' + str(maxage_arr[i]) + ' yrs', color=palette[i],linewidth=1.5)
        
        if show_xlabel:    
            plt.xlabel('Frequency (Hz)', fontsize=40)
        if show_ylabel:
            plt.ylabel('Power Spectrum Density (dB)', fontsize=30)
        plt.title(stages + ' - ' + regions.upper(), fontsize=40)
        plt.xlim([0,20])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('psd_figs/' + stages+'_' + regions + '_plot.png')
        plt.clf()
