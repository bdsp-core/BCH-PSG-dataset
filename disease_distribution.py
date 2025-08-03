import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
from collections import Counter

def compute_percentage(finame,title):
  
  print(finame)
  df = pd.read_csv(finame)
  ID = df['BIDS'].to_numpy()
  
  data = []

  
  for i in range(0,len(ID)):

    try: 
      df_diagosis = pd.read_csv(os.path.join('/home/ayush/Documents/BCH_dataset', str(ID[i]), str(ID[i]) + '_diagnosis.csv') ) #Replace the path with folder where dataset exists
    except pd.errors.EmptyDataError:
      continue
      
    
    if 'Diagnosis' in df_diagosis.columns:
      unique_diagnosis = df_diagosis['Diagnosis'].unique()
    
      for diagnosis in unique_diagnosis:
        data.append({'ID' : ID[i], 'Diagnosis': str(diagnosis)})
  
  result_df = pd.DataFrame(data)
  result_df.to_csv(title.replace(' ', '_')+'_individual_disease_distribution.csv',index=False)
  
  unique_text_list, unique_text_list_num = np.unique(result_df['Diagnosis'].to_numpy(), return_counts=True)
  perc = np.round(unique_text_list_num/len(ID),6)
  

  df_save = pd.DataFrame()
  df_save['Disease Diagnosis'] = np.array(unique_text_list)
  df_save['Rep'] = np.array(unique_text_list_num)
  df_save['Perc'] = np.array(perc)
  
  df_save.to_csv(title.replace(' ', '_')+'_count_disease_distribution.csv',index=False)
  

def process_patient_data(patient_csv, output_csv):
    """
    Function to process patient diagnosis data, map ICD codes to categories, and save the output to a CSV file.

    Parameters:
    - patient_csv: Path to the CSV file containing 'patient ID' and 'diagnosis' columns.
    - mapping_csv: Path to the CSV file containing 'diagnosis' and 'ICD code' columns for mapping non-ICD formatted diagnoses.
    - output_csv: Path to save the output CSV file.
    """
    
    # Load the CSV files
    patient_data = pd.read_csv(patient_csv)  # Contains columns 'patient ID' and 'diagnosis'
    mapping_data = pd.read_csv('Mapping_list.csv')  # Contains columns 'diagnosis' and 'ICD code'

    # Function to categorize ICD code into one of the 22 categories
    def categorize_icd(icd_code):
        if icd_code.startswith('A') or icd_code.startswith('B'):
            return 1  # Infectious diseases
        elif icd_code.startswith('C') or icd_code.startswith('D') and icd_code <= 'D49':
            return 2  # Neoplasms
        elif icd_code.startswith('D') and '50' <= icd_code[1:] <= '89':
            return 3  # Blood disorders
        elif icd_code.startswith('E'):
            return 4  # Endocrine diseases
        elif icd_code.startswith('F'):
            return 5  # Mental disorders
        elif icd_code.startswith('G'):
            return 6  # Nervous system diseases
        elif icd_code.startswith('H') and icd_code <= 'H59':
            return 7  # Eye diseases
        elif icd_code.startswith('H') and '60' <= icd_code[1:] <= '95':
            return 8  # Ear diseases
        elif icd_code.startswith('I'):
            return 9  # Circulatory system diseases
        elif icd_code.startswith('J'):
            return 10  # Respiratory system diseases
        elif icd_code.startswith('K'):
            return 11  # Digestive system diseases
        elif icd_code.startswith('L'):
            return 12  # Skin diseases
        elif icd_code.startswith('M'):
            return 13  # Musculoskeletal diseases
        elif icd_code.startswith('N'):
            return 14  # Genitourinary diseases
        elif icd_code.startswith('O'):
            return 15  # Pregnancy, childbirth, and puerperium
        elif icd_code.startswith('P'):
            return 16  # Perinatal conditions
        elif icd_code.startswith('Q'):
            return 17  # Congenital abnormalities
        elif icd_code.startswith('R'):
            return 18  # Symptoms and abnormal findings
        elif icd_code.startswith('S') or icd_code.startswith('T'):
            return 19  # Injury and external causes
        elif icd_code.startswith('V') or icd_code.startswith('W') or icd_code.startswith('X') or icd_code.startswith('Y'):
            return 20  # External causes of morbidity
        elif icd_code.startswith('Z'):
            return 21  # Health status and services
        elif icd_code.startswith('U'):
            return 22  # Codes for special purposes
        return None  # For unmapped cases

    # Create a dictionary to store patient IDs with their respective categories
    patient_categories = {}
    unmapped = []

    # Process each row of the patient_data
    for _, row in patient_data.iterrows():
        patient_id = row['ID']
        diagnosis = row['Diagnosis']
        
        # Check if the diagnosis follows the ICD format (3 characters)
        #print(diagnosis)
        #print(np.isnan(diagnosis))
        
        if pd.isna(diagnosis):
            continue
        
        
        
        if re.match(r'^[A-Z][0-9]{2}', diagnosis):
            icd_code = diagnosis[:3]
        else:
            # Map the diagnosis using the mapping file
            mapped_row = mapping_data[mapping_data['Term'] == diagnosis]
            if not mapped_row.empty:
                icd_code = mapped_row['ICD_clean'].values[0][:3]
            else:
                icd_code = None
                unmapped.append(diagnosis)
        
        
        # If there's a valid ICD code, categorize it
        if icd_code:
            category = categorize_icd(icd_code)
            if category:
                if patient_id not in patient_categories:
                    patient_categories[patient_id] = [0] * 22  # Initialize 22 categories as 0
                patient_categories[patient_id][category - 1] = 1  # Set the relevant category to 1

    # Create the final DataFrame
    final_data = []
    for patient_id, categories in patient_categories.items():
        final_data.append([patient_id] + categories)

    # Define shortened names for the 22 categories
    category_names = [
        "Infectious", "Oncology", "Hematology", "Endocrinology",
        "Psychiatry", "Neurology", "Ophthalmology", "Otolaryngology (ENT)",
        "Cardiology", "Pulmonology", "Gastroenterology", "Dermatology",
        "Orthopedics", "Urology", "Obstetrics", "Neonatology",
        "Genetics", "Internal Medicine", "Emergency Medicine", "Ext. Morbidity Causes",
        "Misc. Diagnosis", "Special Diagnosis"
    ]

    # Create a DataFrame with appropriate column names
    columns = ['patient ID'] + category_names
    final_df = pd.DataFrame(final_data, columns=columns)

    # Save the output to a CSV file
    final_df.to_csv(output_csv, index=False)
    
    print(len(final_df['patient ID'].to_numpy()))

    print(f"Output saved to '{output_csv}'")
    
    unique_IDarr, unique_IDarr_num = np.unique(unmapped, return_counts=True)
    print(len(unique_IDarr))

    # Count the number of patients in each category
    category_counts = final_df.iloc[:, 1:].sum(axis=0)

    # Sort category counts and corresponding category names in ascending order
    sorted_counts = category_counts.sort_values(ascending=True)
    sorted_category_names = sorted_counts.index

    # Plotting the bar plot in ascending order
    plt.figure(figsize=(12, 8))  # Increase the figure size for better visibility
    bars = plt.bar(sorted_category_names, sorted_counts.values, color='skyblue')

    # Add number on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10)

    # Set ytick labels to diagonal and small font size
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=10)
    
    # Increase font size of axis labels and title for publication quality
    plt.xlabel('Diagnostic Category', fontsize=25)
    plt.ylabel('Number of Patients', fontsize=25)
    plt.title('Number of Patients in Each Diagnostic Category', fontsize=25)
    
    # Add gridlines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot as an image for the paper
    plt.tight_layout()
    plt.savefig('patient_category_plot.png', dpi=300)  # High resolution for publication
    plt.show()
    
compute_percentage(finame='Demographics.csv',title='Total')

process_patient_data('Total_individual_disease_distribution.csv', 'patient_disease_categories.csv')
