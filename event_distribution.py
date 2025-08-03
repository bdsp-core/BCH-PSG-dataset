import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import seaborn as sns
from collections import Counter


def return_counts(df):


  ca_arr = []
  ma_arr = []
  oa_arr = []
  a_arr = []
  ch_arr = []
  mh_arr = []
  oh_arr = []
  h_arr = []
  arousal_arr = []
  lm_arr = []
  rera_arr = []

  ID = df['BIDS'].to_numpy()
  sess = df['Session'].to_numpy()
  
  
  for i in range(0,len(sess)):
    df_annot = pd.read_csv(os.path.join('/home/ayush/Documents/BCH_dataset', str(ID[i]), 'ses-' + str(sess), 'eeg',  str(ID[i]) + str(ID[i]) + '_ses-' + str(sess[i]) + '_eventannotations.csv') )
    
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('centralapnea','Central Apnea')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('apnea','Apnea')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('arousal','Arousal')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('hypopneacentral','Central Hypopnea')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('hypopnea','Hypopnea')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('hypopnea mixed','Mixed Hypopnea')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('hypopneaobstructive','Obstructive Hypopnea')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('limbmvt','Limb Movement')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('mixedapnea','Mixed Apnea')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('obstructiveapnea','Obstructive Apnea')
    df_annot['Standardised Event'] = df_annot['Standardised Event'].replace('rera','RERA')
    
    
    ca_arr.append(df_annot['Standardised Event'].value_counts().get('Central Apnea', 0))
    ma_arr.append(df_annot['Standardised Event'].value_counts().get('Mixed Apnea', 0))
    oa_arr.append(df_annot['Standardised Event'].value_counts().get('Obstructive Apnea', 0))
    a_arr.append(df_annot['Standardised Event'].value_counts().get('Apnea', 0))
    ch_arr.append(df_annot['Standardised Event'].value_counts().get('Central Hypopnea', 0))
    mh_arr.append(df_annot['Standardised Event'].value_counts().get('Mixed Hypopnea', 0))
    oh_arr.append(df_annot['Standardised Event'].value_counts().get('Obstructive Hypopnea', 0))
    h_arr.append(df_annot['Standardised Event'].value_counts().get('Hypopnea', 0))
    arousal_arr.append(df_annot['Standardised Event'].value_counts().get('Arousal', 0))
    lm_arr.append(df_annot['Standardised Event'].value_counts().get('Limb Movement', 0))
    rera_arr.append(df_annot['Standardised Event'].value_counts().get('RERA', 0))
    
    
  return np.array(ca_arr), np.array(ma_arr), np.array(oa_arr), np.array(a_arr), np.array(ch_arr), np.array(mh_arr), np.array(oh_arr), np.array(h_arr), np.array(arousal_arr), np.array(lm_arr), np.array(rera_arr)
  
  
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
  
ca1, ma1, oa1, a1, ch1, mh1, oh1, h1, arousal1, lm1, rera1 = return_counts(df=df_b6)
ca2, ma2, oa2, a2, ch2, mh2, oh2, h2, arousal2, lm2, rera2 = return_counts(df=df_6m1y)
ca3, ma3, oa3, a3, ch3, mh3, oh3, h3, arousal3, lm3, rera3 = return_counts(df=df_1y6y)
ca4, ma4, oa4, a4, ch4, mh4, oh4, h4, arousal4, lm4, rera4 = return_counts(df=df_6y12y)
ca5, ma5, oa5, a5, ch5, mh5, oh5, h5, arousal5, lm5, rera5 = return_counts(df=df_12y18y)
ca6, ma6, oa6, a6, ch6, mh6, oh6, h6, arousal6, lm6, rera6 = return_counts(df=df_above18y)



data = [ca1,ca2,ca3,ca4,ca5,ca6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
#sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
#sns.boxplot(data=data,palette='Set2',showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=14)
plt.yticks(fontsize=12)
plt.title('Distribution of Cental Apnea' , fontsize = 24)
plt.xlabel('Age Group',fontsize=14)
plt.ylabel('#Cental Apnea', fontsize = 14)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Central_Apnea_plot.png')
plt.clf()



data = [ma1,ma2,ma3,ma4,ma5,ma6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
#sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
#sns.boxplot(data=data,palette='Set2',showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=14)
plt.yticks(fontsize=12)
plt.title('Distribution of Mixed Apnea' , fontsize = 24)
plt.xlabel('Age Group',fontsize=14)
plt.ylabel('#Mixed Apnea', fontsize = 14)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Mixed_Apnea_plot.png')
plt.clf()

data = [oa1,oa2,oa3,oa4,oa5,oa6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
#sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
#sns.boxplot(data=data,palette='Set2',showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=14)
plt.yticks(fontsize=12)
plt.title('Distribution of Obstructive Apnea' , fontsize = 24)
plt.xlabel('Age Group',fontsize=14)
plt.ylabel('#Obstructive Apnea', fontsize = 14)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Obstructive_Apnea_plot.png')
plt.clf()

data = [ch1,ch2,ch3,ch4,ch5,ch6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
#sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
#sns.boxplot(data=data,palette='Set2',showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=14)
plt.yticks(fontsize=12)
plt.title('Distribution of Central Hypopnea' , fontsize = 24)
plt.xlabel('Age Group',fontsize=14)
plt.ylabel('#Central Hypopnea', fontsize = 14)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Central_Hypopnea_plot.png')
plt.clf()

data = [mh1,mh2,mh3,mh4,mh5,mh6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
#sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
#sns.boxplot(data=data,palette='Set2',showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=14)
plt.yticks(fontsize=12)
plt.title('Distribution of Mixed Hypopnea' , fontsize = 24)
plt.xlabel('Age Group',fontsize=14)
plt.ylabel('#Mixed Hypopnea', fontsize = 14)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Mixed_Hypopnea_plot.png')
plt.clf()

data = [oh1,oh2,oh3,oh4,oh5,oh6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
#sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
#sns.boxplot(data=data,palette='Set2',showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=14)
plt.yticks(fontsize=12)
plt.title('Distribution of Obstructive Hypopnea' , fontsize = 24)
plt.xlabel('Age Group',fontsize=14)
plt.ylabel('#Obstructive Hypopnea', fontsize = 14)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Obstructive_Hypopnea_plot.png')
plt.clf()

data = [arousal1,arousal2,arousal3,arousal4,arousal5,arousal6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
#sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
#sns.boxplot(data=data,palette='Set2',showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=14)
plt.yticks(fontsize=12)
plt.title('Distribution of Arousal' , fontsize = 24)
plt.xlabel('Age Group',fontsize=14)
plt.ylabel('#Arousal', fontsize = 14)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Arousal_plot.png')
plt.clf()


data = [lm1,lm2,lm3,lm4,lm5,lm6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
#sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
#sns.boxplot(data=data,palette='Set2',showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=14)
plt.yticks(fontsize=12)
plt.title('Distribution of Limb Movement' , fontsize = 24)
plt.xlabel('Age Group',fontsize=14)
plt.ylabel('#Limb Movement', fontsize = 14)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Limb_Movement_plot.png')
plt.clf()


data = [rera1,rera2,rera3,rera4,rera5,rera6]
labels = ['<6mo','6mo-1y','1-6y','6-12y','12-18y','>18y']
#sns.set(style='whitegrid')
plt.figure(figsize=(12,8))
#sns.boxplot(data=data,palette='Set2',showfliers=False)
means = [np.mean(group) for group in data]
plt.plot(range(len(data)), means, marker='o', color='red', linestyle='--', label='Mean Trend')
plt.xticks(ticks=np.arange(len(labels)),labels=labels,fontsize=14)
plt.yticks(fontsize=12)
plt.title('Distribution of RERA' , fontsize = 24)
plt.xlabel('Age Group',fontsize=14)
plt.ylabel('#RERA Movement', fontsize = 14)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('RERA_plot.png')
plt.clf()
