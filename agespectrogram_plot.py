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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

##########################################################################################################
# NEED TO RUN THE SCRIPT psd_save.py first to extact the PSD
##########################################################################################################

def run_model(region,whichstage):
    show_colorbar = False
    show_xlabel = False
    show_ylabel = False
    
    if region=='frontal':
        ch1 = 'F3'
        ch2 = 'F4'
        show_ylabel = True
    elif region=='central':
        ch1 = 'C3'
        ch2 = 'C4'
    elif region=='occipital':
        ch1 = 'O1'
        ch2 = 'O2'
        show_colorbar = True
        
    if whichstage=='REM':
        show_xlabel = True
        
    left = np.load(whichstage+'_' + ch1 + '.npy')
    left_age = np.load(whichstage+'_' + ch1 + '_age.npy')
    right = np.load(whichstage+'_' + ch2 + '.npy')
    right_age = np.load(whichstage+'_' + ch2 + '_age.npy')
    avg_age = (left_age+right_age)/2
    print(np.mean(avg_age-left_age))
    avg_psd = (left+right)/2
    
    psg_corrected = np.zeros((len(left_age),40))
    final_age  = np.zeros((len(left_age),))
    
    ctr = 0

    for i in range(0,len(left_age)):
        if np.min(avg_psd[i,:])!=0:
          psg_corrected[ctr,:] = 10*np.log10(avg_psd[i,:]*1000000000000)
          final_age[ctr] = avg_age[i]
          ctr = ctr + 1
          
    psg_corrected = psg_corrected[0:ctr,:]
    final_age = final_age[0:ctr]


    poly = PolynomialFeatures(degree=2) 
    age_poly_arr = poly.fit_transform((np.array(final_age)).reshape(-1, 1))
    
    
    model = Sequential([
      Input(shape=(age_poly_arr.shape[1],)),
      Dense(50, activation='elu'),
      Dense(50, activation='elu'),
      Dense(40, activation='elu') 
      ])
      
    model.compile(optimizer=Adam(), loss='mse')
    
    model.fit(age_poly_arr, psg_corrected, epochs=10000, batch_size=int(len(final_age)),verbose=0)
    
    pred = np.transpose(model.predict(poly.transform(np.linspace(0,20,30000).reshape(-1, 1))))
    #print(pred.shape)
    
    if stages=='WAKE':
        plt.imshow(pred,cmap='turbo',origin='lower',aspect='auto',vmin=8,vmax=20,extent=(0,20,0,20))
        if show_ylabel:
            plt.ylabel('Freq (Hz)', fontsize=40)
        if show_xlabel:
            plt.xlabel('Age (Years)', fontsize=40)
        plt.title(whichstage + ' - ' + region.upper(), fontsize=35)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout()
        plt.savefig('spect_figs_deg2/' + whichstage+'_' + region + '_plot.png')
        plt.clf()
        
    else:
        plt.imshow(pred,cmap='turbo',origin='lower',aspect='auto',vmin=0,vmax=20,extent=(0,20,0,20))
        plt.title(whichstage + ' - ' + region.upper(), fontsize=35)
        if show_ylabel:
            plt.ylabel('Freq (Hz)', fontsize=40)
        if show_xlabel:
            plt.xlabel('Age (Years)', fontsize=40)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout()
        plt.savefig('spect_figs_deg2/' + whichstage+'_' + region + '_plot.png')
        plt.clf()
    
for stages in ['WAKE','N1','N2','N3','REM']: 
    for regions in ['occipital','frontal','central']:
        run_model(regions,stages)
        
