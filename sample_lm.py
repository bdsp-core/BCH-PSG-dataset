import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd
import numpy as np
import os
import mne
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import auc, roc_curve
from itertools import cycle
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import PrecisionRecallDisplay
import scipy
from scipy.integrate import simpson
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
import h5py
import mne
from mne.filter import filter_data, notch_filter
from scipy.ndimage import label
from mne.time_frequency import psd_array_multitaper
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection


def eeg_filter(eeg, Fs, notch_freq=60., bandpass_low=0.02, bandpass_high=60):
    """
    eeg filter
    """
    
    eeg = eeg.astype(np.float64)
    
    notch_freq = notch_freq  # [Hz]
    bandpass_freq = [bandpass_low, bandpass_high]  # [Hz]
    
    # filter EEG
    if notch_freq is not None:
        eeg = notch_filter(eeg, Fs, notch_freq, verbose=False)
    if bandpass_freq is not None:
        eeg = filter_data(eeg, Fs, bandpass_freq[0], bandpass_freq[1], verbose=False)

    return eeg


def return_filtered_signal(signal):
    
    signal = signal.flatten()
    p5 = np.percentile(signal, 5)
    p95 = np.percentile(signal, 95)
    
    filtered_signal = signal[(signal >= p5) & (signal <= p95)]
    

    mean_filtered = np.mean(filtered_signal)
    std_filtered = np.std(filtered_signal)

    normalized_signal = (signal - mean_filtered) / std_filtered
    
    #print(normalized_signal.shape)
    
    
    filtered_signal = 100*eeg_filter(normalized_signal, Fs=512, notch_freq=60., bandpass_low=0.02, bandpass_high=100)
    
    #plt.plot(filtered_signal)
    #plt.show()
    
    return filtered_signal
    
    
file_path =  '/home/ayush/Documents/BCH_dataset/sub-I0003175866702/ses-1/eeg/sub-I0003175866702_ses-1_task-PSG_eeg.h5'

with h5py.File(file_path, 'r') as f:
    lat = f['signals']['lat'][:].squeeze().astype(np.float64)
    lat = return_filtered_signal(lat)
    
    rat = f['signals']['rat'][:].squeeze().astype(np.float64)
    rat = return_filtered_signal(rat)
    
    limbevent_arr = f['annotations']['limb'][:].squeeze()  
    
    
star_ind = 230
rleg = rat[ int(star_ind*60*200): int((star_ind+40)*60*200)] 
lleg = lat[ int(star_ind*60*200): int((star_ind+40)*60*200)] 
lmevent_arr = limbevent_arr[ int(star_ind*60*200): int((star_ind+40)*60*200)] 

mul = 60*200

start = 0
jump = 10



left_emg_signals = [lleg[int(start*mul):int((start+jump)*mul)],lleg[int((start+jump)*mul):int((start+2*jump)*mul)],lleg[int((start+2*jump)*mul):int((start+3*jump)*mul)],lleg[int((start+3*jump)*mul):int((start+4*jump)*mul)]]  
right_emg_signals = [rleg[int(start*mul):int((start+jump)*mul)],rleg[int((start+jump)*mul):int((start+2*jump)*mul)],rleg[int((start+2*jump)*mul):int((start+3*jump)*mul)],rleg[int((start+3*jump)*mul):int((start+4*jump)*mul)]] 
binary_arrays = [lmevent_arr[int(start*mul):int((start+jump)*mul)],lmevent_arr[int((start+jump)*mul):int((start+2*jump)*mul)],lmevent_arr[int((start+2*jump)*mul):int((start+3*jump)*mul)],lmevent_arr[int((start+3*jump)*mul):int((start+4*jump)*mul)]] 


min_y_left = np.min(np.array([np.quantile(np.array(left_emg_signals),0.001),np.quantile(np.array(right_emg_signals),0.001)]))
max_y_left = np.max(np.array([np.quantile(np.array(left_emg_signals),0.999),np.quantile(np.array(right_emg_signals),0.999)]))
min_y_right = np.min(np.array([np.quantile(np.array(left_emg_signals),0.001),np.quantile(np.array(right_emg_signals),0.001)]))
max_y_right = np.max(np.array([np.quantile(np.array(left_emg_signals),0.999),np.quantile(np.array(right_emg_signals),0.999)]))


time = np.linspace(0, 2, int(10*60*200))


# Downsample the data for better performance (Adjust the factor based on need)
ds_factor = 1  # Downsample factor
time_ds = time[::ds_factor]
left_emg_signals_ds = [sig[::ds_factor] for sig in left_emg_signals]
right_emg_signals_ds = [sig[::ds_factor] for sig in right_emg_signals]
binary_arrays_ds = [bin_arr[::ds_factor] for bin_arr in binary_arrays]

# Function to create LineCollection for binary array
def create_line_collection(time, binary_array):
    segments = []
    for idx in range(len(binary_array) - 1):
        if binary_array[idx] == 0:
            segments.append([(time[idx], 0.5), (time[idx + 1], 0.5)])
        else:
            segments.append([(time[idx], 0.5), (time[idx + 1], 0.5)])
    
    line_segments = LineCollection(segments, colors=['white' if b == 0 else 'blue' for b in binary_array[:-1]], 
                                   linewidths=[1 if b == 0 else 10 for b in binary_array[:-1]])
    return line_segments

# Create figure and axes
fig, axs = plt.subplots(12, 1, figsize=(10, 6), sharex=True)  # Reduce figure height to 8
plt.subplots_adjust(hspace=-1.0)  # Reduce the space between subplots


for i in range(4):
    # Plot left EMG (top signal for each combination)
    axs[i * 3].plot(time_ds, left_emg_signals_ds[i], color='black')
    #axs[i * 3].set_ylabel(f'Left EMG {i+1}')
    axs[i * 3].set_ylim(min_y_left,max_y_left)
    
    # Plot right EMG (middle signal for each combination)
    axs[i * 3 + 1].plot(time_ds, right_emg_signals_ds[i], color='black')
    #axs[i * 3 + 1].set_ylabel(f'Right EMG {i+1}')
    axs[i * 3 + 1].set_ylim(min_y_right,max_y_right)
    
    # Plot binary line with varying styles using LineCollection (bottom signal for each combination)
    lc = create_line_collection(time_ds, binary_arrays_ds[i])
    axs[i * 3 + 2].add_collection(lc)
    axs[i * 3 + 2].set_xlim(time_ds[0], time_ds[-1])
    #axs[i * 3 + 2].set_ylabel(f'Array Line {i+1}')

    # Remove bounding boxes (axes spines), x-axis labels, gridlines, and ticks
    for j in range(3):
        axs[i * 3 + j].spines['top'].set_visible(False)
        axs[i * 3 + j].spines['right'].set_visible(False)
        axs[i * 3 + j].spines['left'].set_visible(False)
        axs[i * 3 + j].spines['bottom'].set_visible(False)
        axs[i * 3 + j].set_xticks([])
        axs[i * 3 + j].xaxis.set_ticklabels([])
        axs[i * 3 + j].grid(False)
        axs[i * 3 + j].set_yticks([])

# Show the plot
plt.tight_layout()

#plt.show()
plt.savefig('LM_plot.png')

  