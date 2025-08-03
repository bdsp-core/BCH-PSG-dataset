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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_respiratory_segments_with_labels(df, fs=10.0, savefilename='resp_plot_segmented_labeled'):
    """
    Plots 4 time-divided subplots with stacked black signals and labeled y-axis lines.
    Includes arousal bar (top), respiratory event bar (bottom), and color legend at figure bottom.
    """

    # Define signal layout and labels
    signals = ['airflow', 'breathing_trace', 'chest', 'abd', 'spo2']
    signal_labels = ['Airflow', 'Pres', 'Chest', 'Abd', 'SpO2']
    n_signals = len(signals)

    total_len = len(df)
    seg_len = total_len // 4
    time = np.arange(total_len) / fs

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    # Color map for respiratory events
    event_colors = {
        1: 'blue',     # obstructive
        2: 'green',    # central
        3: 'cyan',     # mixed
        4: 'magenta',  # hypopnea
        5: 'red'       # RERA
    }

    vertical_gap = 12  # space between signals in a subplot

    for i in range(4):
        ax = axes[i]
        start = i * seg_len
        end = (i + 1) * seg_len if i < 3 else total_len
        t = np.arange(end - start) / fs

        y_base = (n_signals + 2) * vertical_gap  # total stack height for each subplot

        # --- Plot arousal bar ---
        arousal = df['EEG_arousals'].iloc[start:end].values
        arousal_y = y_base
        ax.fill_between(t, arousal_y, arousal_y + 2, where=arousal > 0,
                        color='black', step='pre')

        ax.text(-5, arousal_y + 1, 'Arousal', va='center', ha='right', fontsize=9)

        # --- Plot signals ---
        for j, sig in enumerate(signals):
            if sig == 'spo2':
                y = df[sig].iloc[start:end].values
                y = (y - np.mean(y)) / np.std(y) * 2  # z-score and scale to about plus minus 5
            else:
                y = df[sig].iloc[start:end].values
            offset = y_base - (j + 1) * vertical_gap
            ax.plot(t, y + offset, color='black', linewidth=0.8)
            ax.text(-5, offset, signal_labels[j], va='center', ha='right', fontsize=9)

        # --- Plot respiratory event bar ---
        event_y = y_base - (n_signals + 1) * vertical_gap
        events = df['y'].iloc[start:end].values
        for val in sorted(set(events)):
            if val == 0:
                continue
            mask = events == val
            ax.fill_between(t, event_y, event_y + 2, where=mask,
                            color=event_colors.get(int(val), 'gray'), step='pre')

        ax.text(-5, event_y + 1, 'Event', va='center', ha='right', fontsize=9)

        ax.set_ylim(event_y - 5, y_base + 5)
        ax.set_yticks([])
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)
        if i == 3:
            ax.set_xlabel('Time (s)')

    # Add legend for event labels at bottom
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for label, color in zip(
            ['Obstructive', 'Central', 'Mixed', 'Hypopnea', 'RERA'],
            ['blue', 'green', 'cyan', 'magenta', 'red']
        )
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0.02, 0.05, 1, 1])
    plt.savefig(f"{savefilename}.png", dpi=300)
    plt.close()



 
def return_filtered_signal(signal):
    
    signal = signal.flatten()
    p5 = np.percentile(signal, 5)
    p95 = np.percentile(signal, 95)
    
    filtered_signal = signal[(signal >= p5) & (signal <= p95)]
    

    mean_filtered = np.mean(filtered_signal)
    std_filtered = np.std(filtered_signal)

    normalized_signal = (signal - mean_filtered) / std_filtered
    
    #print(normalized_signal.shape)
    
    filtered_signal = mne.filter.filter_data(normalized_signal, sfreq=200, l_freq=None, h_freq=10,verbose=False)
    
    #plt.plot(filtered_signal)
    #plt.show()
    
    return filtered_signal
 
file_path =  '/home/ayush/Documents/BCH_dataset/sub-I0003175866702/ses-1/eeg/sub-I0003175866702_ses-1_task-PSG_eeg.h5'

  
with h5py.File(file_path, 'r') as f:
    abd = f['signals']['abd'][:].squeeze().astype(np.float64)
    abd = return_filtered_signal(abd)
    abd = scipy.signal.resample(abd, int(10*len(abd)/200))
    
    chest = f['signals']['chest'][:].squeeze().astype(np.float64)
    chest = return_filtered_signal(chest)
    chest = scipy.signal.resample(chest, int(10*len(chest)/200))
    
    spo2 = f['signals']['sao2'][:].squeeze().astype(np.float64)
    
    '''
    max_val = np.max(spo2)
    if max_val > 100:
        spo2 = spo2 - (max_val - 100)
    
    spo2 = np.clip(spo2, 0, None)
    '''
    
    print(max(spo2))
    print(min(spo2))
    #spo2 = return_filtered_signal(spo2)
    spo2 = scipy.signal.resample(spo2, int(10*len(spo2)/200))
    
    pres = f['signals']['pressure'][:].squeeze().astype(np.float64)
    pres = return_filtered_signal(pres)
    pres = scipy.signal.resample(pres, int(10*len(pres)/200))
    
    flow = f['signals']['airflow'][:].squeeze().astype(np.float64)
    flow = return_filtered_signal(flow)
    flow = scipy.signal.resample(flow, int(10*len(flow)/200))
    
    stage_raw = f['annotations']['stage'][:].squeeze()
    arousal_arr = f['annotations']['arousal'][:].squeeze()
    respevent_arr = f['annotations']['resp'][:].squeeze()


# Downsample to 10 Hz (assuming 200 Hz originally)
downsample_factor = 20
stage_down = stage_raw[::downsample_factor]
arousal_down = arousal_arr[::downsample_factor]    
resp_down = respevent_arr[::downsample_factor]

#Wanted code -> 0=NaN, 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake
#Original code -> sleep_mapping = {"N1": 0, "N2": 1, "N3": 2, "N4": 2, "REM": 3, "WAKE": 4, "UNSCORED":9}
stage_remap = {
    0: 3,  # N1 ? N1
    1: 2,  # N2 ? N2
    2: 1,  # N3 ? N3
    3: 4,  # REM ? REM
    4: 5,  # WAKE ? WAKE
    9: 0   # UNSCORED ? NaN class
}
stage_down_mapped = np.vectorize(stage_remap.get)(stage_down)



# Wanted code-> 0=no event, 1=obst a, 2=cen a, 3=mix a, 4=hyp, 5=RERA
# orignal code -> {'None': 0, 'centralapnea': 1, 'apnea': 2, 'hypopneacentral': 3, 'hypopnea': 4, 'hypopnea mixed': 5, 'hypopneaobstructive': 6, 'mixedapnea': 7, 'obstructiveapnea': 8, 'rera': 9}
resp_remap = {
    0: 0,  # None ? no event
    1: 2,  # centralapnea ? central apnea
    2: 2,  # apnea ? central apnea
    3: 4,  # hypopneacentral ? hypopnea
    4: 4,  # hypopnea ? hypopnea
    5: 4,  # hypopnea mixed ? hypopnea
    6: 4,  # hypopneaobstructive ? hypopnea
    7: 3,  # mixedapnea ? mixed apnea
    8: 1,  # obstructiveapnea ? obstructive apnea
    9: 5   # rera ? RERA
}
resp_down_mapped = np.vectorize(resp_remap.get)(resp_down)


#for stt_val in [330, 340, 345, 100, 150, 200, 370, 10, 50, 220]:
for stt_val in [330]:
    print(stt_val)
    stt = int(stt_val*60*10)
    ett = int(stt+12000)   
        
        
    df = pd.DataFrame()
    df['sleep_stages'] = stage_down_mapped[stt:ett].astype(float)
    df['spo2'] = spo2[stt:ett]
    df['abd'] = abd[stt:ett]
    df['chest'] = chest[stt:ett]
    df['breathing_trace'] = -1*pres[stt:ett]
    df['airflow'] = -1*flow[stt:ett]
    df['EEG_arousals'] = arousal_down[stt:ett].astype(float)
    df['y'] = resp_down_mapped[stt:ett].astype(float)
    df['yp'] = resp_down_mapped[stt:ett].astype(float)
    
    plot_respiratory_segments_with_labels(df, fs=10.0,savefilename='new_resp_plot_'+str(stt_val))
