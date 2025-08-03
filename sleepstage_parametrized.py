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
  
  wake = []
  rem = []
  n1 = []
  n2 = []
  n3 = []
  agedays_final = []
  ctr = 0
  
  for i in range(0,len(sess)):
    df_sleepannot = pd.read_csv( os.path.join('/home/ayush/Documents/BCH_dataset', str(ID[i]), 'ses-' + str(sess), 'eeg',  str(ID[i]) + str(ID[i]) + '_ses-' + str(sess[i]) + '_sleepannotations.csv')    )
    
    df_sleepannot['sleep_stage'] = df_sleepannot['sleep_stage'].replace('N4','N3')
    
    stages_of_interest = ['WAKE' , 'REM' , 'N1' , 'N2', 'N3']
    
    filtered_df = df_sleepannot[df_sleepannot['sleep_stage'].isin(stages_of_interest)]
    
    stage_percentages = filtered_df['sleep_stage'].value_counts(normalize=True) * 100
    
    stage_percentages = stage_percentages.reindex(stages_of_interest,fill_value=0)
    
    ctr = ctr + 1
    
    wake.append(stage_percentages[0]/100)
    rem.append(stage_percentages[1]/100)
    n1.append(stage_percentages[2]/100)
    n2.append(stage_percentages[3]/100)
    n3.append(stage_percentages[4]/100)
    agedays_final.append(agedays[i])
      
  
  
  poly = PolynomialFeatures(degree=2) 
  age_poly_arr = poly.fit_transform((np.array(agedays_final)/365).reshape(-1, 1))
  
  y = np.transpose(np.array([wake,n1,n2,n3,rem]))
  
  
  
  
  model = Sequential([
  Input(shape=(age_poly_arr.shape[1],)),
  Dense(5, activation='softmax') 
  ])
  
  #print(model.summary())
  
  # Compile the model
  model.compile(optimizer=Adam(), loss='categorical_crossentropy')
  
  # Train the model
  model.fit(age_poly_arr, y, epochs=10000, batch_size=15695,verbose=1)
  #model.fit(age_poly_arr, y, epochs=1, batch_size=15695,verbose=1)
  
  
  pred = model.predict(poly.transform(np.linspace(0,20,30000).reshape(-1, 1)))

  wake_pred = 100*pred[:,0]
  n1_pred = 100*pred[:,1]
  n2_pred = 100*pred[:,2]
  n3_pred = 100*pred[:,3]
  rem_pred = 100*pred[:,4]
  
  
  df_stages = pd.DataFrame({'age': np.linspace(0,20,30000),'wake': np.array(wake_pred),'n1': np.array(n1_pred),'n2': np.array(n2_pred), 'n3': np.array(n3_pred), 'rem': np.array(rem_pred)})



  
  fig, ax = plt.subplots(figsize=(10, 6))
  
  ax.bar(df_stages['age'], df_stages['wake'], label='W',color='gold',align='edge')
  ax.bar(df_stages['age'], df_stages['n1'], bottom=df_stages['wake'], label='N1', color='lightblue',align='edge')
  ax.bar(df_stages['age'], df_stages['n2'], bottom=df_stages['wake'] + df_stages['n1'], label='N2',color='blue',align='edge')
  ax.bar(df_stages['age'], df_stages['n3'], bottom=df_stages['wake'] + df_stages['n1'] + df_stages['n2'], label='N3', color='darkblue',align='edge')
  ax.bar(df_stages['age'], df_stages['rem'], bottom=df_stages['wake'] + df_stages['n1'] + df_stages['n2'] + df_stages['n3'], label='R', color='purple',align='edge')
  
  
  ax.set_ylim(-0.01, 100.01 )
  
  ax.set_xlim(df_stages['age'].min()-0.02, df_stages['age'].max()+0.02)
  
  #ax.legend_.remove() if ax.legend_ else None
  
  
  
  # Customizing the plot
  ax.set_xlabel('Age in years', fontsize=16)
  ax.set_ylabel('Percentage Sleep Stage', fontsize=16)
  
  plt.title('Sleep Stages Distribution by Age' , fontsize = 20)
  
  #ax.legend(loc='lower left', ncol=1, bbox_to_anchor=(0, 0), mode='expand', borderaxespad=0, frameon=False)
  ax.legend(loc='upper right', 
          bbox_to_anchor=(1.10, 1),  # Position outside the plot
          ncol=1,  # Set to 1 for vertical arrangement
          frameon=True,  # Bounding box visible
          facecolor='white',  # Background color of legend
          edgecolor='black',  # Border color of legend
          framealpha=1)  # Solid, opaque bounding box

  
  #ax.legend()
  
  
  # Show the plot
  plt.tight_layout()
  plt.savefig('stacked_bar_plot_smooth_nn_cont_age_final.png', dpi=300)
  plt.show()
    
   
    

	
	
compute_percentage(finame='Demographics.csv',title='Entire dataset')

	
