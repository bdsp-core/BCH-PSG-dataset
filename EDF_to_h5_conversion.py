import pandas as pd
import os
import json
import numpy as np
from collections import Counter
import mne
import scipy
from scipy.integrate import simpson
import concurrent.futures
import h5py
from scipy.signal import medfilt
mne.set_log_level('ERROR') 


def get_signal(data,channel,channel_names):
    if channel in channel_names:
        if channel in ['SaO2','EtCO2','TcCO2']:
            return 100*abs(data[channel_names.index(channel), :])
        else:
            return data[channel_names.index(channel), :]
    else:
        return np.zeros(data.shape[1])


def convert_to_h5(start_ind,end_ind):

        main_folder = '/home/ayush/Documents/BCH_dataset'
        
        finame='Demographics.csv'
        
        df = pd.read_csv(finame)
        ID = df['subID'].to_numpy()
        sess = df['Session'].to_numpy()
        agedays = df['AgeDays'].to_numpy()
        
        name_counts = {}
        
        for i in range(start_ind,end_ind):

            try:
	            prefix = f"{ID[i]}_ses-{sess[i]}_"
	            arranged_dir = '/home/ayush/Documents/BCH_h5_arranged'
	            
	            # Check if any folder starting with prefix already exists
	            already_done = any(fname.startswith(prefix) for fname in os.listdir(arranged_dir))
	            if already_done:
	                print(f"Skipping: {prefix} already processed.")
	                continue
	            


	              
	            for fi in os.listdir(os.path.join(main_folder,ID[i],'ses-'+str(sess[i]),'eeg')):

	              if fi.endswith('_sleepannotations.csv'):
	                df_sleepannot = pd.read_csv(os.path.join(main_folder,ID[i],'ses-'+str(sess[i]),'eeg',fi))
	                df_sleepannot['sleep_stage'] = df_sleepannot['sleep_stage'].replace('N4','N3')
	                filtered_df_epoch = df_sleepannot[df_sleepannot['epoch'] > 0]

	                sleep_annot_stage = filtered_df_epoch['sleep_stage'].to_numpy()
	                sleep_annot_start = filtered_df_epoch['sample_stamp_start'].to_numpy()
	                sleep_annot_end = filtered_df_epoch['sample_stamp_end'].to_numpy()

	                

	            for fi in os.listdir(os.path.join(main_folder,ID[i],'ses-'+str(sess[i]),'eeg')):
	          
	              if fi.endswith('.edf'):
	              
	                annot_path = os.path.join('/home/ayush/Documents/BCH_dataset', str(ID[i]), 'ses-' + str(sess[i]), 'eeg',  str(ID[i]) + str(ID[i]) + '_ses-' + str(sess[i]) + '_eventannotations.csv')
	                df_eventannot = pd.read_csv(annot_path)
	                
	                mask_Chest_Abdomen = df_eventannot['Comment'].str.contains(r'\bChest\b|\bAbdomen\b', case=True, na=False)
	                mask_CHEST_ABD = df_eventannot['Comment'].str.contains(r'\bCHEST\b|\bABD\b', case=True, na=False)

	                mask_nap_int = df_eventannot['Comment'].str.contains(r'\bNAP\s?\(INT\)', case=False, na=False)
	                
	                if mask_Chest_Abdomen.any() and not mask_CHEST_ABD.any():
	                    raw = mne.io.read_raw_edf(os.path.join(main_folder,ID[i],'ses-'+str(sess[i]),'eeg',fi), include = ['C3','C4','O1','O2','F3','F4','E1','LOC','E2','ROC','M1','A1','M2','A2','CHIN1','CHIN2','ECGL','ECG-LA','ECGR','ECG-RA','Chest','Abdomen','FLOW','Pressure', 'DIF6', 'DIF3+', 'DIF4+','DC8','DC7','DC5','SNORE','DIF5+','DC9','DC4','DC10','DC3','LAT1','LLEG+','LAT2','LLEG-','RAT1','RLEG+','RAT2','RLEG-'], preload=True, verbose='ERROR')
	                else:                  
	                    raw = mne.io.read_raw_edf(os.path.join(main_folder,ID[i],'ses-'+str(sess[i]),'eeg',fi), include = ['C3','C4','O1','O2','F3','F4','E1','LOC','E2','ROC','M1','A1','M2','A2','CHIN1','CHIN2','ECGL','ECG-LA','ECGR','ECG-RA','CHEST','DIF1+','ABD','DIF2+','FLOW','Pressure', 'DIF6', 'DIF3+', 'DIF4+','DC8','DC7','DC5','SNORE','DIF5+','DC9','DC4','DC10','DC3','LAT1','LLEG+','LAT2','LLEG-','RAT1','RLEG+','RAT2','RLEG-'], preload=True, verbose='ERROR')

	                #print(len(raw))

	                signal_start = sleep_annot_start[0] - 1

	                for var in range(len(sleep_annot_end)-1,-1,-1):
	                    if sleep_annot_end[var]<=len(raw):
	                            signal_end =  sleep_annot_end[var]
	                            end_var = var
	                            break

	                sleeparray = np.repeat(sleep_annot_stage[0:end_var+1], 30 * 200)  
	                sleep_mapping = {"N1": 0, "N2": 1, "N3": 2, "N4": 2, "REM": 3, "WAKE": 4}
	                numerical_sleeparray = np.vectorize(lambda x: sleep_mapping.get(x, 9))(sleeparray).astype(int)

	                
	                sfreq = raw.info['sfreq']  # Sampling frequency
	                start_time = signal_start / sfreq  # Convert start index to time (in seconds)
	                end_time = signal_end / sfreq      # Convert end index to time (in seconds)

	                # Crop the raw object in place (this modifies the original 'raw' object)
	                raw.crop(tmin=start_time, tmax=end_time, include_tmax=False)


	                
	                arousalarray = np.zeros((int((signal_end - signal_start)*(200/sfreq)),))
	                resparray = np.full(int((signal_end - signal_start)*(200/sfreq)), 'None', dtype='<U30') 
	                limbarray = np.zeros((int((signal_end - signal_start)*(200/sfreq)),))

	                dataindex_array = df_eventannot['DataIndex'].to_numpy() 
	                event_array = df_eventannot['Standardised Event'].to_numpy()
	                duration_array = df_eventannot['Duration'].to_numpy()

	                for j in range(0,len(duration_array)):
	                    if np.isnan(duration_array[j])==False:
	                        if event_array[j]=='arousal':
	                            if dataindex_array[j]>=signal_start and dataindex_array[j]<=signal_end:
	                                arousal_start = int(np.floor( (dataindex_array[j]-signal_start) *200/sfreq))
	                                arousal_end = int(np.ceil( (dataindex_array[j]-signal_start)*200/sfreq) + int(duration_array[j]*200))
	                                arousalarray[arousal_start:arousal_end+1] = 1
	                        
	                        if event_array[j]=='limbmvt':
	                            if dataindex_array[j]>=signal_start and dataindex_array[j]<=signal_end:
	                                limb_start = int(np.floor( (dataindex_array[j]-signal_start) *200/sfreq))
	                                limb_end = int(np.ceil( (dataindex_array[j]-signal_start)*200/sfreq) + int(duration_array[j]*200))
	                                limbarray[limb_start:limb_end+1] = 1

	                        if event_array[j] in {'centralapnea','apnea','hypopneacentral','hypopnea', 'hypopnea mixed','hypopneaobstructive','mixedapnea','obstructiveapnea','rera'}:
	                            if dataindex_array[j]>=signal_start and dataindex_array[j]<=signal_end:
	                                resp_start = int(np.floor( (dataindex_array[j]-signal_start) *200/sfreq))
	                                resp_end = int(np.ceil( (dataindex_array[j]-signal_start)*200/sfreq) + int(duration_array[j]*200))
	                                resparray[resp_start:resp_end+1] = event_array[j]

	                resp_mapping = {'None': 0, 'centralapnea': 1, 'apnea': 2, 'hypopneacentral': 3, 'hypopnea': 4, 'hypopnea mixed': 5, 'hypopneaobstructive': 6, 'mixedapnea': 7, 'obstructiveapnea': 8, 'rera': 9}
	                numerical_resparray = np.vectorize(resp_mapping.get)(resparray).astype(int)



	                # Define renaming dictionary
	                rename_dict_master = {
	                    'A1': 'M1',
	                    'A2': 'M2',  
	                    'LOC': 'E1',
	                    'ROC': 'E2',
	                    'ECG-LA': 'ECGL',
	                    'ECG-RA': 'ECGR',
	                    'DIF1+': 'CHEST',
	                    'Chest': 'CHEST',
	                    'Abdomen': 'ABD',
	                    'DIF2+': 'ABD',
	                    'DIF4+': 'FLOW',
	                    'DC8': 'PLETH',
	                    'DC5': 'EtCO2',
	                    'DIF5+': 'SNORE',
	                    'DC9': 'CAPNO',
	                    'DC4': 'CFLOW',
	                    'DC10': 'CPRES',
	                    'LLEG+': 'LAT1',
	                    'LLEG-': 'LAT2',
	                    'RLEG+': 'RAT1',
	                    'RLEG-': 'RAT2',
	                    'DC7': 'SaO2',
	                    'DC3': 'TcCO2',
	                }

	                # Create rename_dict with only available channels in raw.ch_names
	                rename_dict = {key: rename_dict_master[key] for key in rename_dict_master if key in raw.ch_names}

	                # Apply renaming
	                raw.rename_channels(rename_dict)
	                
	                
	                raw.resample(sfreq=200, npad='auto')
	        
	                data = raw.get_data() 
	                channel_names = raw.ch_names

	        
	                signals = {}
	        
	                reference_M1_M2 = 'M1' if 'M1' in channel_names else 'A1'
	                reference_M2_A2 = 'M2' if 'M2' in channel_names else 'A2'
	        
	        
	                
	                signals['C3-M2'] = get_signal(data,'C3',channel_names) - get_signal(data,'M2',channel_names)
	                signals['C4-M1'] = get_signal(data,'C4',channel_names) - get_signal(data,'M1',channel_names)
	                signals['O1-M2'] = get_signal(data,'O1',channel_names) - get_signal(data,'M2',channel_names)
	                signals['O2-M1'] = get_signal(data,'O2',channel_names) - get_signal(data,'M1',channel_names)
	                signals['F3-M2'] = get_signal(data,'F3',channel_names) - get_signal(data,'M2',channel_names)
	                signals['F4-M1'] = get_signal(data,'F4',channel_names) - get_signal(data,'M1',channel_names)
	                signals['E1-M2'] = get_signal(data,'E1',channel_names) - get_signal(data,'M2',channel_names)
	                signals['E2-M1'] = get_signal(data,'E2',channel_names) - get_signal(data,'M1',channel_names)
	                signals['CHIN1-CHIN2'] = get_signal(data,'CHIN1',channel_names) - get_signal(data,'CHIN2',channel_names)
	                signals['ECG'] = get_signal(data,'ECGL',channel_names) - get_signal(data,'ECGR',channel_names)
	                signals['CHEST'] = get_signal(data,'CHEST',channel_names) 
	                signals['ABD'] = get_signal(data,'ABD',channel_names) 
	                signals['AIRFLOW'] = get_signal(data,'FLOW',channel_names) 

	                if 'DIF6' in channel_names:
	                    signals['Pressure'] = get_signal(data,'DIF6',channel_names) 
	                else:
	                    if mask_nap_int.any():
	                        signals['Pressure'] = get_signal(data,'Pressure',channel_names) 
	                    else:
	                        signals['Pressure'] = get_signal(data,'DIF3+',channel_names) 

	                
	                signals['PWAVE'] = get_signal(data,'PLETH',channel_names) 
	                signals['SaO2'] = get_signal(data,'SaO2',channel_names) 
	                signals['EtCO2'] = get_signal(data,'EtCO2',channel_names) 
	                signals['TcCO2'] = get_signal(data,'TcCO2',channel_names) 
	                signals['SNORE'] = get_signal(data,'SNORE',channel_names) 
	                signals['CAPNO'] = get_signal(data,'CAPNO',channel_names)
	                signals['CFLOW'] = get_signal(data,'CFLOW',channel_names)  
	                signals['CPRES'] = get_signal(data,'CPRES',channel_names) 

	                if 'LAT1' in raw.ch_names and 'LAT2' in raw.ch_names:
	                    signals['lat'] = get_signal(data,'LAT1',channel_names) - get_signal(data,'LAT2',channel_names)
	                else:
	                    signals['lat'] = get_signal(data,'LAT1',channel_names)

	                if 'RAT1' in raw.ch_names and 'RAT2' in raw.ch_names:
	                    signals['rat'] = get_signal(data,'RAT1',channel_names) - get_signal(data,'RAT2',channel_names)
	                else:
	                    signals['rat'] = get_signal(data,'RAT1',channel_names)



	                
	                with h5py.File(os.path.join('/home/ayush/Documents/BCH_h5_arranged',fi[:-3]+'h5'), 'w') as f:
	                    f.attrs['sampling_rate'] = 200
	                    f.attrs['unit_voltage'] = 'uV'
	                    f.attrs['AgeinDays'] = agedays[i]
	                    group_signals = f.create_group('signals')
	                    
	                    for key, value in signals.items():
	                        if key in ['SaO2', 'EtCO2', 'TcCO2']:
	                            #print(value.shape)
	                            #print(key)
	                            #print(max(value))
	                            #print(min(value))
	                            group_signals.create_dataset(key.lower(), data=abs(value), shape=(len(value), 1), maxshape=(len(value), 1), dtype='float32', compression="gzip")
	                        else:
	                            #print(value.shape)
	                            group_signals.create_dataset(key.lower(), data=value*1000000, shape=(len(value), 1), maxshape=(len(value), 1), dtype='float32', compression="gzip")
	                          
	                    group_annotations = f.create_group('annotations')
	                    group_annotations.create_dataset('stage', data=numerical_sleeparray, shape=(len(value), 1), maxshape=(len(value), 1), compression="gzip")  # Storing as fixed-length string
	                    group_annotations.create_dataset('arousal', data=arousalarray, dtype='int8', shape=(len(value), 1), maxshape=(len(value), 1), compression="gzip")  # Binary 0 or 1
	                    group_annotations.create_dataset('resp', data=numerical_resparray, shape=(len(value), 1), maxshape=(len(value), 1), compression="gzip")  # Strings up to 30 chars
	                    group_annotations.create_dataset('limb', data=limbarray, dtype='int8', shape=(len(value), 1), maxshape=(len(value), 1), compression="gzip")  # Binary 0 or 1
	                    

	                    if ( len(numerical_sleeparray) != len(value) or len(arousalarray) != len(value) or len(numerical_resparray) != len(value) or len(limbarray) != len(value)):
	                        print( str(i) + '   ' + fi[:-3]+'h5' )
	                        '''
	                        print(len(numerical_sleeparray))
	                        print(len(arousalarray))
	                        print(len(numerical_resparray))
	                        print(len(limbarray))
	                        print(len(value))
	                        '''
     
            
            except:                
            	print('Fail: ' + os.path.join(ID[i],'ses-'+str(sess[i]),'eeg')) 
            
            
            

ranges = [(i, min(i + 500, 15695)) for i in range(0, 15695, 500)]

    
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(convert_to_h5, start, end) for start, end in ranges]
    
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()  # This will also raise any exception that occurred in the thread
        except Exception as e:
            print(f"Generated an exception: {e}")





  
                  
