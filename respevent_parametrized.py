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

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.special import softmax
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam



def compute_percentage(finame,title):

  print(finame)
  df = pd.read_csv(finame)
  ID = df['BIDS'].to_numpy()
  sess = df['Session'].to_numpy()
  agedays = df['AgeDays'].to_numpy()
  
  centralap = []
  centralhyp = []
  mixed = []
  obstructiveap = []
  obstructivehyp = []
  rera = []
  ageyrs_decimal = []
  ctr = 0
  
  for i in range(0,len(sess)):
    df_annot = pd.read_csv( os.path.join('/home/ayush/Documents/BCH_dataset', str(ID[i]), 'ses-' + str(sess), 'eeg',  str(ID[i]) + str(ID[i]) + '_ses-' + str(sess[i]) + '_eventannotations.csv')    )
    
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
    
    stages_of_interest = ['Central Apnea' , 'Central Hypopnea', 'Mixed Apnea' , 'Mixed Hypopnea', 'Obstructive Apnea' , 'Obstructive Hypopnea', 'RERA']
    
    filtered_df = df_annot[df_annot['Standardised Event'].isin(stages_of_interest)]
    
    stage_percentages = filtered_df['Standardised Event'].value_counts(normalize=True)
    
    stage_percentages = stage_percentages.reindex(stages_of_interest,fill_value=0)
    
    if sum(stage_percentages)>=0.99 and sum(stage_percentages)<=1.01:
    
      ctr = ctr + 1
      
      centralap.append(stage_percentages[0])
      centralhyp.append(stage_percentages[1])
      mixed.append(stage_percentages[2]+stage_percentages[3])
      obstructiveap.append(stage_percentages[4])
      obstructivehyp.append(stage_percentages[5])
      rera.append(stage_percentages[6])
      ageyrs_decimal.append(agedays[i]/365)
      
  
  
  poly = PolynomialFeatures(degree=2) 
  age_poly_arr = poly.fit_transform((np.array(ageyrs_decimal)).reshape(-1, 1))
  
  y = np.transpose(np.array([centralap,centralhyp,mixed,obstructiveap,obstructivehyp,rera]))
  
  print(age_poly_arr.shape)
  print(y.shape)
  print(ctr)
  
  
  
  
  model = Sequential([
  Input(shape=(age_poly_arr.shape[1],)),
  Dense(6, activation='softmax') 
  ])
  
  #print(model.summary())
  
  # Compile the model
  model.compile(optimizer=Adam(), loss='categorical_crossentropy')
  
  # Train the model
  model.fit(age_poly_arr, y, epochs=10000, batch_size=15695,verbose=1)
  
  
  pred = model.predict(poly.transform(np.linspace(0,20,30000).reshape(-1, 1)))

  centralap_pred = 100*pred[:,0]
  centralhyp_pred = 100*pred[:,1]
  mixed_pred = 100*pred[:,2]
  obstructiveap_pred = 100*pred[:,3]
  obstructivehyp_pred = 100*pred[:,4]
  rera_pred = 100*pred[:,5]
  
  
  df_stages = pd.DataFrame({'age': np.linspace(0,20,30000),'centralap': np.array(centralap_pred),'centralhyp': np.array(centralhyp_pred),'mixed': np.array(mixed_pred),'obstructiveap': np.array(obstructiveap_pred), 'obstructivehyp': np.array(obstructivehyp_pred), 'rera': np.array(rera_pred)})



  
  fig, ax = plt.subplots(figsize=(10, 6))
  
  ax.bar(df_stages['age'], df_stages['centralap'], label='CA',color='lightgreen',align='edge')
  ax.bar(df_stages['age'], df_stages['centralhyp'], bottom=df_stages['centralap'], label='CH', color='green',align='edge')
  ax.bar(df_stages['age'], df_stages['mixed'], bottom=df_stages['centralap'] + df_stages['centralhyp'], label='MA/MH', color='purple',align='edge')
  ax.bar(df_stages['age'], df_stages['obstructiveap'], bottom=df_stages['centralap'] + df_stages['centralhyp'] + df_stages['mixed'], label='OA',color='lightblue',align='edge')
  ax.bar(df_stages['age'], df_stages['obstructivehyp'], bottom=df_stages['centralap'] + df_stages['centralhyp'] + df_stages['mixed'] + df_stages['obstructiveap'], label='OH',color='blue',align='edge')
  ax.bar(df_stages['age'], df_stages['rera'], bottom=df_stages['centralap'] + df_stages['centralhyp'] + df_stages['mixed'] + df_stages['obstructiveap'] + df_stages['obstructivehyp'], label='RERA',color='red',align='edge')
  
  
  #ax.set_xticks([])  # Remove x-tick marks
  #ax.set_xticklabels([])  # Remove x-tick labels
  #ax.set_yticks([])  # Remove y-tick marks
  #ax.set_yticklabels([])  # Remove y-tick labels
  
  ax.set_ylim(-0.01, 100.01 )
  
  ax.set_xlim(df_stages['age'].min()-0.02, df_stages['age'].max()+0.02)
  
  #ax.legend_.remove() if ax.legend_ else None
  
  
  
  # Customizing the plot
  ax.set_xlabel('Age in years', fontsize=14)
  ax.set_ylabel('Percentage Respiratory Event', fontsize=14)
  
  plt.title('Respiratory Event Distribution by Age' , fontsize = 18)
  
  ax.legend(loc='upper right', 
          bbox_to_anchor=(1.15, 1),  # Position outside the plot
          ncol=1,  # Set to 1 for vertical arrangement
          frameon=True,  # Bounding box visible
          facecolor='white',  # Background color of legend
          edgecolor='black',  # Border color of legend
          framealpha=1)  # Solid, opaque bounding box
  
  #ax.legend()
  
  
  # Show the plot
  plt.tight_layout()
  plt.savefig('stacked_bar_plot_resp_age_deg2_divide.png', dpi=300)
  #plt.savefig('stacked_bar_plot_' + title.replace(' ', '_')+'.png')
  plt.show()
    
   
    

	
	
compute_percentage(finame='Demographics.csv',title='Entire dataset')

	
