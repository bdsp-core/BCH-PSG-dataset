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
    
def spectrogram(signal, Fs, signaltype=None, epoch_time=30, epoch_step_time=30, decibel=True, fmin=0.02, fmax=60, bandwidth=None, adaptive=True, n_jobs=1):
    """
    Inputs:
    signal: 1d numpy array of signal (time domain)
    Fs: sampling frequency
    signaltype: keywords/shortcuts (see code below, selects bandwith based on keyword)
    epoch_time: window-length in seconds
    epoch_step_time: stepsize in seconds
    decibel: boolean, if result shall be return in decibel (default True)
    fmin: minimum frequency of interest
    fmax: maximum frequency of interest
    bandwidth: multi-taper bandwidth parameter
    adaptive: (see MNE description. True=more accurate but slow)
    n_jobs: parallel jobs.
    Returns:
    # specs.shape = (#epoch, #channel, #freq)
    # freq.shape = (#freq,)
    """
    
    # segment
    epoch_size = int(round(epoch_time*Fs))
    epoch_step = int(round(epoch_step_time*Fs))
    start_ids = np.arange(0, signal.shape[1]-epoch_size+1, epoch_step)
    seg_ids = list(map(lambda x:np.arange(x,x+epoch_size), start_ids))
    signal_segs = signal[:,seg_ids].transpose(1,0,2)  # signal_segs.shape=(#epoch, #channel, Tepoch)
    if 0:
        print(signal_segs.shape)
    # compute spectrogram

    if bandwidth is None:
        if signaltype == 'eeg':       
            NW = 10.
            bandwidth = NW*2./epoch_time
        elif signaltype == 'resp_effort':
            NW = 1
            bandwidth = NW/epoch_time
        else:
            raise ValueError("Unexpected signaltype! ")

    # experimenting values with toy data:
    # bandwidth = 1
    # half_nbw = 0.55
    # bandwidth = half_nbw / (epoch_time  * Fs / (2. * Fs))
    # print(bandwidth)

    # this is how half nbw is computed in code:
    # n_times = signal_segs.shape[-1]
    # half_nbw = float(bandwidth) * n_times / (2. * sfreq)
    # n_tapers_max = int(2 * half_nbw)

    specs, freq = psd_array_multitaper(signal_segs, Fs, fmin=fmin, fmax=fmax, adaptive=adaptive, low_bias=True, verbose='ERROR', bandwidth=bandwidth, normalization='full', n_jobs=n_jobs)

    if decibel:
        specs = 10*np.log10(specs)
    
    return specs, freq, signal_segs
    
file_path = '/home/ayush/Documents/BCH_dataset/sub-I0003175866702/ses-1/eeg/sub-I0003175866702_ses-1_task-PSG_eeg.h5' #change for different h5

eeg_channels = ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1']
central_eeg_channels = ['c3-m2', 'c4-m1']
frontal_eeg_channels = [ 'f3-m2', 'f4-m1']
occipital_eeg_channels = [ 'o1-m2', 'o2-m1']
    
region_specs = {
    'central': [],
    'frontal': [],
    'occipital': []
}

with h5py.File(file_path, 'r') as f:
    sampling_rate = f.attrs['sampling_rate']
    for ch in eeg_channels:
        eeg = f['signals'][ch][:].squeeze().astype(np.float64)
        eeg = eeg_filter(eeg, sampling_rate)
        specs,freq,eeg_segs = spectrogram(np.transpose(np.reshape(eeg,(len(eeg),1))),Fs = sampling_rate, signaltype='eeg', epoch_time=2, epoch_step_time=1, bandwidth=2, fmin=0, fmax=20)
        specs = specs.squeeze().T
    
        if ch in central_eeg_channels:
            region_specs['central'].append(specs)
        elif ch in frontal_eeg_channels:
            region_specs['frontal'].append(specs)
        elif ch in occipital_eeg_channels:
            region_specs['occipital'].append(specs)

# Compute average spectrogram per region
avg_specs = {
    region: np.mean(np.stack(specs, axis=0), axis=0)  # shape: (freqs, time)
    for region, specs in region_specs.items()
}


fig, axs = plt.subplots(4, 1, figsize=(12, 10))

with h5py.File(file_path, 'r') as f:
    stage_raw = f['annotations']['stage'][:].squeeze()
    arousal_arr = f['annotations']['arousal'][:].squeeze()


# Downsample to 1 Hz (assuming 200 Hz originally)
downsample_factor = 200
stage_down = stage_raw[::downsample_factor]
arousal_down = arousal_arr[::downsample_factor]
time_hours = np.arange(len(stage_down)) / 3600  # 1 Hz ? 1 value per second

stage_remap = {4: 4, 3: 3, 0: 2, 1: 1, 2: 0}  # WAKE at top
stage_down_mapped = np.vectorize(stage_remap.get)(stage_down)

# --- Plot hypnogram ---
axs[0].step(time_hours, stage_down_mapped, where='post', color='black', linewidth=0.75)
axs[0].set_yticks([0, 1, 2, 3, 4])
axs[0].set_yticklabels(['N3','N2','N1','REM','WAKE'])
axs[0].set_ylabel('Hypnogram', fontsize=12)
axs[0].set_xlim([0, time_hours[-1]])

# --- Highlight REM segments in red ---
rem_mask = (stage_down == 3)
rem_regions, _ = label(rem_mask)

for r in range(1, rem_regions.max() + 1):
    idx = np.where(rem_regions == r)[0]
    axs[0].plot(time_hours[idx], [3] * len(idx), color='red', linewidth=1.25)

# --- Highlight arousal regions in orange ---
arousal_mask = (arousal_down == 1)
arousal_regions, _ = label(arousal_mask)

for r in range(1, arousal_regions.max() + 1):
    idx = np.where(arousal_regions == r)[0]
    axs[0].axvspan(time_hours[idx[0]], time_hours[idx[-1]], color='orange', alpha=0.4)



regions = ['frontal', 'central', 'occipital']
step_sec = 1  # step size in seconds
titles = ['Frontal', 'Central', 'Occipital']

for i, region in enumerate(regions):
    spec = avg_specs[region]
    n_steps = spec.shape[1]
    time_hours = np.arange(n_steps) * step_sec / 3600  # seconds to hours
    
    im = axs[i+1].imshow(spec, cmap='turbo', origin='lower', aspect='auto',
                         extent=(time_hours[0], time_hours[-1], freq.min(), freq.max()),
                         vmin=0, vmax=20)
    axs[i+1].set_ylim([0, 20])
    axs[i+1].set_ylabel(titles[i], fontsize=14)
    
    # Only bottom plot gets xlabel
    if i == 2:
        axs[i+1].set_xlabel('Time (hours)', fontsize=14)

# Add one shared colorbar
fig.subplots_adjust(right=0.88)  # leave space for colorbar
cbar_ax = fig.add_axes([0.90, 0.13, 0.015, 0.74])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='Power (dB)')

plt.tight_layout(rect=[0, 0, 0.88, 1])  # adjust for colorbar space
#plt.show()
plt.savefig('hypnogram_spectrogram_plot.png', dpi=300, bbox_inches='tight')





