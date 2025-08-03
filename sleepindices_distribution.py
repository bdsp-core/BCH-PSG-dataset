import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
from collections import Counter
import seaborn as sns
from sleepindices import compute_sfi,sleep_indices_stages



def return_sleepindices(df):

    ID = df['BIDS'].to_numpy()
    sess = df['Session'].to_numpy()
    
    sfiarr = []
    hours_sleep_arr = []
    hours_psg_arr = []
    sleep_efficiency_arr = []
    waso_arr = []
    sleep_latency_arr = []
    r_latency_arr = []
    
    for i in range(0,len(sess)):
        
        df_sleepannot = pd.read_csv( os.path.join('/home/ayush/Documents/BCH_dataset', str(ID[i]), 'ses-' + str(sess), 'eeg',  str(ID[i]) + str(ID[i]) + '_ses-' + str(sess[i]) + '_sleepannotations.csv')    )
        df_sleepannot['sleep_stage'] = df_sleepannot['sleep_stage'].replace('N4','N3')
        
        filtered_df_epoch = df_sleepannot[df_sleepannot['epoch'] > 0]
        
        sampf = int((df_sleepannot['sample_stamp_end'].to_numpy()[2]-df_sleepannot['sample_stamp_end'].to_numpy()[1])/30)
        
        
        events_of_interest = ['N1', 'N2', 'N3', 'WAKE', 'REM']
        filtered_df = filtered_df_epoch[filtered_df_epoch['sleep_stage'].isin(events_of_interest)]
        
        sleep_stage = filtered_df['sleep_stage'].to_numpy()
        stamp_start = filtered_df['sample_stamp_start'].to_numpy()
        stamp_end = filtered_df['sample_stamp_end'].to_numpy()
        
        
        signal_len = int(stamp_end[-1] - stamp_start[0] + 1)
        
        #print( str(ID[i]) + '_ses-' + str(sess[i]) + ' ' + str(stamp_end[-1]) + ' ' + str(stamp_start[0]) + ' ' + str(signal_len) )
        
        annots = np.zeros(signal_len,)
        
        for j in range(0,len(stamp_start)):
            if sleep_stage[j]=='N3':
                annots[int(j*sampf*30):int((j+1)*sampf*30)] = 1
            if sleep_stage[j]=='N2':
                annots[int(j*sampf*30):int((j+1)*sampf*30)] = 2
            if sleep_stage[j]=='N1':
                annots[int(j*sampf*30):int((j+1)*sampf*30)] = 3
            if sleep_stage[j]=='REM':
                annots[int(j*sampf*30):int((j+1)*sampf*30)] = 4
            if sleep_stage[j]=='WAKE':
                annots[int(j*sampf*30):int((j+1)*sampf*30)] = 5
                
        sfi = compute_sfi(annots,sampf)
        hours_sleep, hours_psg, sleep_efficiency, perc_r, perc_n1, perc_n2, perc_n3, waso, sleep_latency, r_latency = sleep_indices_stages(annots,sampf)
        
        sfiarr.append(sfi)
        hours_sleep_arr.append(hours_sleep)
        hours_psg_arr.append(hours_psg)
        sleep_efficiency_arr.append(sleep_efficiency)
        waso_arr.append(waso)
        sleep_latency_arr.append(sleep_latency)
        r_latency_arr.append(r_latency)
        
    
    return [sfiarr, hours_sleep_arr, hours_psg_arr, sleep_efficiency_arr, waso_arr, sleep_latency_arr, r_latency_arr]
    

def save_plot(data,title,ylabel,savename):
    labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
    sns.set(style='whitegrid')
    plt.figure(figsize=(12,8))
    #sns.boxplot(data=data,palette='Set2')
    sns.boxplot(data=data,boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),  
            whiskerprops=dict(color='black', linewidth=2),  
            capprops=dict(color='black', linewidth=2),  
            medianprops=dict(color='black', linewidth=3),
            flierprops=dict(markerfacecolor='black', markeredgecolor='black'),showfliers=False)
    means = [np.nanmean(group) for group in data]
    plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', linewidth=4, markersize=10, label='Mean Trend')
    plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=30)
    plt.yticks(fontsize=30)
    plt.title(title, fontsize = 40)
    plt.xlabel('Age Group',fontsize=30)
    plt.ylabel(ylabel, fontsize = 30)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig('sleepindices_'+savename+'.png')
    plt.clf()


df_main = pd.read_csv('/path/to/Demographics.csv')
bins = [0, 181, 366, 2191, 4381, 6571, float('inf')]
labels = ['<6mo', '6mo-1y', '1-6y', '6-12y', '12-18y', '>18y']
df_main['AgeGroup'] = pd.cut(df['AgeDays'], bins=bins, labels=labels, right=False)

df_b6 = df_main[df_main['AgeGroup'] == '<6mo']
df_6m1y = df_main[df_main['AgeGroup'] == '6mo–1y']
df_1y6y = df_main[df_main['AgeGroup'] == '1–6y']
df_6y12y = df_main[df_main['AgeGroup'] == '6–12y']
df_12y18y = df_main[df_main['AgeGroup'] == '12–18y']
df_above18y = df_main[df_main['AgeGroup'] == '>18y']

  
sfi_b6, hrs_sleep_b6, hrs_psg_b6, sleep_eff_b6, waso_b6, sleep_lat_b6, r_lat_b6 = return_sleepindices(df=df_b6)
sfi_6m1y, hrs_sleep_6m1y, hrs_psg_6m1y, sleep_eff_6m1y, waso_6m1y, sleep_lat_6m1y, r_lat_6m1y = return_sleepindices(df=df_6m1y)
sfi_1y6y, hrs_sleep_1y6y, hrs_psg_1y6y, sleep_eff_1y6y, waso_1y6y, sleep_lat_1y6y, r_lat_1y6y = return_sleepindices(df=df_1y6y)
sfi_6y12y, hrs_sleep_6y12y, hrs_psg_6y12y, sleep_eff_6y12y, waso_6y12y, sleep_lat_6y12y, r_lat_6y12y = return_sleepindices(df=df_6y12y)
sfi_12y18y, hrs_sleep_12y18y, hrs_psg_12y18y, sleep_eff_12y18y, waso_12y18y, sleep_lat_12y18y, r_lat_12y18y = return_sleepindices(df=df_12y18y)
sfi_above18y, hrs_sleep_above18y, hrs_psg_above18y, sleep_eff_above18y, waso_above18y, sleep_lat_above18y, r_lat_above18y = return_sleepindices(df=df_above18y)


save_plot(data = [sfi_b6,sfi_6m1y,sfi_1y6y,sfi_6y12y,sfi_12y18y,sfi_above18y], title = 'Sleep Fragmentation Index' ,ylabel = 'SFI per hour',savename='sfi_boxplot')

save_plot(data = [hrs_sleep_b6,hrs_sleep_6m1y,hrs_sleep_1y6y,hrs_sleep_6y12y,hrs_sleep_12y18y,hrs_sleep_above18y], title = 'Recorded Sleep Duration' ,ylabel = 'Recorded Sleep Duration (hrs)',savename='sleep_hours_boxplot')

save_plot(data = [hrs_psg_b6,hrs_psg_6m1y,hrs_psg_1y6y,hrs_psg_6y12y,hrs_psg_12y18y,hrs_psg_above18y], title = 'PSG Duration' ,ylabel = 'PSG Duration (Hrs)',savename='psg_hours_boxplot')
  
save_plot(data = [sleep_eff_b6,sleep_eff_6m1y,sleep_eff_1y6y,sleep_eff_6y12y,sleep_eff_12y18y,sleep_eff_above18y], title = 'Sleep Efficiency' ,ylabel = 'Sleep Efficiency',savename='sleep_efficiency_boxplot')

save_plot(data = [waso_b6,waso_6m1y,waso_1y6y,waso_6y12y,waso_12y18y,waso_above18y], title = 'Wake After Sleep Onset' ,ylabel = 'WASO (minutes)',savename='waso_boxplot')

save_plot(data = [sleep_lat_b6,sleep_lat_6m1y,sleep_lat_1y6y,sleep_lat_6y12y,sleep_lat_12y18y,sleep_lat_above18y], title = 'Sleep Latency' ,ylabel = 'Sleep Latency (minutes)',savename='sleep_latency_boxplot')

save_plot(data = [r_lat_b6,r_lat_6m1y,r_lat_1y6y,r_lat_6y12y,r_lat_12y18y,r_lat_above18y], title = 'REM Latency' ,ylabel = 'REM Latency (minutes)',savename='rem_latency_boxplot')

