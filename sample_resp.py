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

def nassi_breathing_plot(d, fs,savefilename):
    ''' "d" is a DataFrame with the follwoing columns:
        sleep_stages    - Sleep staging with mapping: 0=NaN, 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake
        spo2            - Pulse oximetry
        abd             - RIP at abdomen
        chest           - RIP at chest 
        breathing_trace - preferrably nasal pressure (PTAf), but could also be flow measured during CPAP  (CFLOW)
        airflow         - Flow measured via thermistor

        EEG_arousals    - EEG Arousal labels 
        y               - expert resp labels with mapping: 0=no event, 1=obst a, 2=cen a, 3=mix a, 4=hyp, 5=RERA
        yp              - CAISR resp label (same mapping as "y")

    '''


    nrow = 4
    sleep_stages = d.sleep_stages.values   
    patient_asleep = np.logical_and(sleep_stages>0, sleep_stages<5)
    # define the ids each row
    row_ids = np.array_split(np.arange(len(d)), nrow)
    row_ids.reverse()

    # set figure, and plooting vars
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    row_height = 50
    
    # set plotting signals
    plot_traces = ['spo2', 'abd', 'chest', 'breathing_trace']
    signal_color = ['r', 'purple', 'g', 'k']
    offsets = [-15, 0, 0, 7.5]
    lws = [.5, .5, .5, .3]
    # add airflow, if availible 
    if not all(d.airflow==0): 
        plot_traces += ['airflow']
        signal_color += ['b']
        offsets += [12]
        lws += [.3]
    # add arousals
    plot_traces += ['EEG_arousals']
    signal_color += ['k']
    offsets += [offsets[-1]]
    lws += [1]
    # platinum arousals
    if 'arousal_platinum' in d.columns:
        plot_traces += ['arousal_platinum']
        signal_color += ['r']
        offsets += [offsets[-1]]
        lws += [1]


    # PLOT SIGNALS
    for trace, color, offset, lw in zip(plot_traces, signal_color, offsets, lws):
        # color all wake parts red
        ww = ~patient_asleep
        if 'bad_signal' in d.columns:
            bad = d.bad_signal == 1
        else:
            bad = np.zeros(len(d)).astype(bool)
        sleep = np.array(d[trace])
        sleep[ww] = np.nan
        sleep[bad] = np.nan
        wake = np.array(d[trace])
        wake[patient_asleep] = np.nan
        wake[bad] = np.nan
        bad_sig = np.array(d[trace])
        bad_sig[~bad] = np.nan

        if trace == 'spo2':
            wake[np.less(wake, 70, where=np.isfinite(wake))] = np.nan
            sleep[np.less(sleep, 70, where=np.isfinite(sleep))] = np.nan
            bad_sig[np.less(sleep, 70, where=np.isfinite(sleep))] = np.nan
            lower, upper = 85, 95 #np.nanmin(sleep), np.nanmax(sleep)
            wake = (wake-lower) / (upper-lower) * 10
            sleep = (sleep-lower) / (upper-lower) * 10
            bad_sig = (bad_sig-lower) / (upper-lower) * 10
        elif trace in ['EEG_arousals', 'arousal_platinum']:
            sleep = np.array(d[trace].values) * 10
            sleep[sleep==0] = np.nan
            wake = np.array(d[trace].values) * 10
            wake[wake==0] = np.nan
            bad_sig = np.array(d[trace].values) * 10
            bad_sig[bad_sig==0] = np.nan
        else:
            wake[np.greater(wake, 10, where=np.isfinite(wake))] = np.nan
            sleep[np.greater(sleep, 10, where=np.isfinite(sleep))] = np.nan
            bad_sig[np.greater(bad_sig, 10, where=np.isfinite(bad_sig))] = np.nan
            wake[np.less(wake, -10, where=np.isfinite(wake))] = np.nan
            sleep[np.less(sleep, -10, where=np.isfinite(sleep))] = np.nan
            bad_sig[np.less(bad_sig, -10, where=np.isfinite(bad_sig))] = np.nan
            wake = wake*1.5
            sleep = sleep*1.5
            bad_sig = bad_sig*1.5

        for ri in range(nrow):
            # plot trace black & wake red
            ax.plot(sleep[row_ids[ri]]+ri*row_height + offset, c=color, lw=lw)
            ax.plot(wake[row_ids[ri]]+ri*row_height + offset, c=color, lw=lw, alpha=0.5)
            ax.plot(bad_sig[row_ids[ri]]+ri*row_height + offset, c=color, lw=lw, alpha=0.5)

    # PLOT LABELS
    yyy = ['y']
    labs = ['Label', 'expert 2', 'expert 3'][:len(yyy)]
    #yyy += ['yp']
    #labs += ['caisr']
    if 'resp-h3_platinum' in d.columns: 
        yyy += ['resp-h3_platinum']
        labs += ['platinum']
    label_color = [None, 'b', 'g', 'c', 'm', 'r']
    for yi, label_tag in enumerate(yyy):    
        # get labels from DF
        labels = d[label_tag].values  
        # replace all zeros by nan's
        labels = labels.astype('float')
        labels[(labels==0)] = float('nan')

        # label offset
        offset = -10-yi
        
        # run over each plot row
        for ri in range(nrow):
            # plot tech annonation
            ax.axhline(ri*row_height-(yi+1)+offset, c=[0.5,0.5,0.5], ls='--', lw=0.5)  # gridline
            loc = 0

            # group all labels and plot them
            for i, j in groupby(labels[row_ids[ri]]):
                len_j = len(list(j))
                if not np.isnan(i) and label_color[int(i)] is not None:
                    # make event alpha=0.4 if >50% during sleep
                    a = .4 if np.sum(patient_asleep[row_ids[ri]][list(range(loc, loc+len_j))] == 1) < ((loc+len_j)-loc) / 2 else 1
                    ax.plot([loc, loc+len_j], [ri*row_height-(yi+1)+offset]*2, c=label_color[int(i)], lw=2, alpha=a)
                loc += len_j

    ### construct legend box ###
    # split line
    row_len = len(d) // nrow
    ax.plot([0, row_len], [-38]*2, c='k', lw=1)

    # event types
    event_types = ['obstructive', 'central', 'mixed', 'hypopnea', 'RERA']
    for i, (color, e_type) in enumerate(zip(label_color[1:], event_types)):
        if i < 3:
            x = int(row_len * 0.017) + int(row_len * 0.125) * i
            y = -45
        else:
            x = int(row_len * 0.08) + int(row_len * 0.125) * (i - 3)
            y = -60
        ax.plot([x, x+30*fs], [y]*2, c=color, lw=4)
        ax.text(x+15*fs, y-3, e_type, fontsize=12, ha='center', va='top')

    # signal legend
    #legend_tags = ['NPT/CFLOW', 'Resp. Effort', 'SpO2']
    #signal_color = ['k', 'g', 'r']
    legend_tags = ['Airflow','Pres', 'Abd', 'Chest','SpO2']
    signal_color = ['b','k','purple', 'g', 'r']
    '''
    if not all(d.airflow==0): 
        legend_tags = ['Airflow'] + legend_tags
        signal_color = ['b'] + signal_color
    '''
    legend_tags = ['Arousals'] + legend_tags
    signal_color = ['k'] + signal_color
    for i, (color, trace) in enumerate(zip(signal_color, legend_tags)):
        x = int(row_len * 0.92)
        y = -45 - 10*i
        if 'arousal' not in trace.lower():
            ax.plot([x-55*fs, x-30*fs], [y]*2, c=color, lw=2)
        else:
            ax.plot([x-35*fs, x-30*fs], [y]*2, c=color, lw=3)
        ax.text(x-20*fs, y, trace, fontsize=10, ha='left', va='center')

    # label legend
    for i, label in enumerate(labs):
        x = int(row_len * 0.6)
        y = -45 - 7.5*i
        ax.plot([x-50*fs, x-30*fs], [y]*2, '--', c='k', lw=1)
        ax.text(x-20*fs, y, label, fontsize=10, ha='left', va='center')
                
    # plot layout setup
    ax.set_xlim([0, max([len(x) for x in row_ids])])
    ax.axis('off')
    #blk = f'\n40min (ind{d.index[0]}-{d.index[-1]})'
    #title = 'Respiratory Plots'
    #ax.set_title(title)
    plt.tight_layout()

    # save the figure
    #fig.savefig('Resp_plot.jpg', dpi=900)
    fig.savefig(savefilename+'.png', dpi=900)
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
 
file_path = '/home/ayush/Documents/BCH_dataset/sub-I0003175866702/ses-1/eeg/sub-I0003175866702_ses-1_task-PSG_eeg.h5'

  
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
    
    nassi_breathing_plot(df, fs=10.0,savefilename='resp_plot_'+str(stt_val))
