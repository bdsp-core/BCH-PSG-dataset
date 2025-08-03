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

def compute_percentage(df,title):

  ID = df['BIDS'].to_numpy()
  sess = df['Session'].to_numpy()
  
  wake = []
  rem = []
  n1 = []
  n2 = []
  n3 = []
  
  
  for i in range(0,len(sess)):
    df_sleepannot = pd.read_csv( os.path.join('/home/ayush/Documents/BCH_dataset', str(ID[i]), 'ses-' + str(sess), 'eeg',  str(ID[i]) + str(ID[i]) + '_ses-' + str(sess[i]) + '_sleepannotations.csv') )
    
    df_sleepannot['sleep_stage'] = df_sleepannot['sleep_stage'].replace('N4','N3')
    
    stages_of_interest = ['WAKE' , 'REM' , 'N1' , 'N2', 'N3']
    
    filtered_df = df_sleepannot[df_sleepannot['sleep_stage'].isin(stages_of_interest)]
    
    stage_percentages = filtered_df['sleep_stage'].value_counts(normalize=True) * 100
    
    stage_percentages = stage_percentages.reindex(stages_of_interest,fill_value=0)
    
    if (stage_percentages[0] + stage_percentages[1] + stage_percentages[2] + stage_percentages[3] + stage_percentages[4])>=99 and (stage_percentages[0] + stage_percentages[1] + stage_percentages[2] + stage_percentages[3] + stage_percentages[4])<=101:
    
      wake.append(stage_percentages[0])
      rem.append(stage_percentages[1])
      n1.append(stage_percentages[2])
      n2.append(stage_percentages[3])
      n3.append(stage_percentages[4])
    
  return wake,rem,n1,n2,n3
	
	
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
	

	
w1,r1,n11,n21,n31 = compute_percentage(df=df_b6,title='0-6 months')
w2,r2,n12,n22,n32 = compute_percentage(df=df_6m1y,title='6-12 months')
w3,r3,n13,n23,n33 = compute_percentage(df=df_1y6y,title='1-6 years')
w4,r4,n14,n24,n34 = compute_percentage(df=df_6y12y,title='6-12 years')
w5,r5,n15,n25,n35 = compute_percentage(df=df_12y18y,title='12-18 years')
w6,r6,n16,n26,n36 = compute_percentage(df=df_above18y,title='>18 years')


data = [w1,w2,w3,w4,w5,w6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
sns.boxplot(data=data, 
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),  
            whiskerprops=dict(color='black', linewidth=2),  
            capprops=dict(color='black', linewidth=2),  
            medianprops=dict(color='black', linewidth=3),
            flierprops=dict(markerfacecolor='black', markeredgecolor='black'),showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', linewidth=4, markersize=10, label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=30)
plt.yticks(fontsize=30)
plt.title('WAKE stage proportion' , fontsize = 50)
plt.xlabel('Age Group',fontsize=35)
plt.ylabel('%', fontsize = 50)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('sleepstage_wake.png')
plt.clf()

data = [n11,n12,n13,n14,n15,n16]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
sns.boxplot(data=data, 
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),  
            whiskerprops=dict(color='black', linewidth=2),  
            capprops=dict(color='black', linewidth=2),  
            medianprops=dict(color='black', linewidth=3),
            flierprops=dict(markerfacecolor='black', markeredgecolor='black'),showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', linewidth=4, markersize=10, label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=30)
plt.yticks(fontsize=30)
plt.title('N1 stage proportion' , fontsize = 50)
plt.xlabel('Age Group',fontsize=35)
plt.ylabel('%', fontsize = 50)
sns.despine()
plt.tight_layout()
plt.savefig('sleepstage_N1.png')
plt.clf()


data = [n21,n22,n23,n24,n25,n26]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
sns.boxplot(data=data, 
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),  
            whiskerprops=dict(color='black', linewidth=2),  
            capprops=dict(color='black', linewidth=2),  
            medianprops=dict(color='black', linewidth=3),
            flierprops=dict(markerfacecolor='black', markeredgecolor='black'),showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', linewidth=4, markersize=10, label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=30)
plt.yticks(fontsize=30)
plt.title('N2 stage proportion' , fontsize = 50)
plt.xlabel('Age Group',fontsize=35)
plt.ylabel('%', fontsize = 50)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('sleepstage_N2.png')
plt.clf()

data = [n31,n32,n33,n34,n35,n36]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
sns.boxplot(data=data, 
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),  
            whiskerprops=dict(color='black', linewidth=2),  
            capprops=dict(color='black', linewidth=2),  
            medianprops=dict(color='black', linewidth=3),
            flierprops=dict(markerfacecolor='black', markeredgecolor='black'),showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', linewidth=4, markersize=10, label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=30)
plt.yticks(fontsize=30)
plt.title('N3 stage proportion' , fontsize = 50)
plt.xlabel('Age Group',fontsize=35)
plt.ylabel('%', fontsize = 50)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('sleepstage_N3.png')
plt.clf()


data = [r1,r2,r3,r4,r5,r6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
sns.boxplot(data=data, 
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),  
            whiskerprops=dict(color='black', linewidth=2),  
            capprops=dict(color='black', linewidth=2),  
            medianprops=dict(color='black', linewidth=3),
            flierprops=dict(markerfacecolor='black', markeredgecolor='black'),showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', linewidth=4, markersize=10, label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=30)
plt.yticks(fontsize=30)
plt.title('REM stage proportion' , fontsize = 50)
plt.xlabel('Age Group',fontsize=35)
plt.ylabel('%', fontsize = 50)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('sleepstage_REM.png')
plt.clf()
	
